local image      = require 'image'
local paths      = require 'paths'
local t          = require 'datasets/transforms'
local ffi        = require 'ffi'
local str_utils  = require 'utils/str_utils'
local flow_utils = require 'utils/flow_utils'

local M = {}
local TOMDataset = torch.class('TOM.TOMDataset', M)
torch.setdefaulttensortype('torch.FloatTensor')

function TOMDataset:readList(file_path)
   local img_info = {}
   local counter   = 0
   for name in io.lines(file_path) do
       counter = counter + 1
       img_info[counter] = name
   end
   print(string.format('Totaling %d images in split %s', counter, self.split))
   return img_info
end

function TOMDataset:__init(opt, split)
   self.opt = opt
   self.split = split
   if split == 'train' then 
      self.img_list = paths.concat(opt.data_dir, opt.train_list)
      self.dir      = paths.concat(opt.data_dir, 'train/Images') 
   elseif split == 'val' then
      self.img_list = paths.concat(opt.data_dir, opt.val_list)
      self.dir      = paths.concat(opt.data_dir, 'val/Images')   
   end

   assert(paths.filep(self.img_list), 'filenames does not exist: ' .. self.img_list)
   assert(paths.dirp(self.dir),       'directory does not exist: ' .. self.dir)

   self.img_info = self:readList(self.img_list)
   print(string.format('Dataset filenames: %s',       self.img_list))
   print(string.format('Dataset image directory: %s', self.dir))
   print(string.format('Image size H*W = %d*%d\n',    self.opt.scale_h, self.opt.scale_h))
end

function TOMDataset:rotateFlow(flow, angle)
  local flow_rot = image.rotate(flow, angle)
  local fu = torch.mul(flow_rot[2], math.cos(-angle)) - torch.mul(flow_rot[1], math.sin(-angle)) 
  local fv = torch.mul(flow_rot[2], math.sin(-angle)) + torch.mul(flow_rot[1], math.cos(-angle))
  flow_rot[2]:copy(fu)
  flow_rot[1]:copy(fv)
  return flow_rot
end

function TOMDataset:get(i)
   local sc_w   = self.opt.scale_w 
   local sc_h   = self.opt.scale_h
   local crop_w = self.opt.crop_w
   local crop_h = self.opt.crop_h
   if self.opt.data_aug and self.split == 'train' then -- random rescale size
       sc_w = torch.random(crop_w , sc_w * 1.05) 
       sc_h = torch.random(crop_h , sc_h * 1.05) 
   end

   local path_tar  = self.img_info[i]
   local path_base = str_utils.splitext(path_tar)
   local path_ref  = path_base .. '_ref.jpg'
   local path_tar  = path_base .. '.jpg' 
   local path_mask = path_base .. '_mask.png'
   local path_rho  = path_base .. '_rho.png'
   local path_flow = path_base .. '_flow.flo'

   local image_ref = self:_loadImage(paths.concat(self.dir, path_ref), 3)
   local image_tar = self:_loadImage(paths.concat(self.dir, path_tar), 3)

   local mask = self:_loadImage(paths.concat(self.dir, path_mask), 1):float():div(255)
   local rho  = self:_loadImage(paths.concat(self.dir, path_rho),  1):float():div(255)

   local flow = flow_utils.loadShortFlowFile(paths.concat(self.dir, path_flow))
   flow:resizeAs(image_ref)

   if path_tar:find('_cplx') then 
       -- Complex object have a large maganitude of refractive flow, 
       -- we set a smaller weight for them to blance the training
       flow[3]:fill(0.5)
   else
       flow[3]:fill(1)
   end

   -- Check if need rescale or crop
   local _, in_h, in_w = unpack(image_ref:size():totable())
   local need_scale  = (in_h ~= sc_h or in_w ~= sc_w)
   local need_aug    = self.opt.data_aug and (self.split == 'train' or self.split=='val')
   local need_flip   = need_aug and (torch.uniform(0,1) > 0.5)
   local need_rotate = need_aug and self.opt.data_rot and self.split == 'train'
   local need_crop   = (sc_h ~= crop_h or sc_w ~= crop_w) and self.split == 'train'

   if need_scale then --print('Image Need scale')
       image_ref = image.scale(image_ref, sc_w, sc_h, 'bilinear')
       image_tar = image.scale(image_tar, sc_w, sc_h, 'bilinear')
       mask      = image.scale(mask,      sc_w, sc_h, 'simple')
       rho       = image.scale(rho,       sc_w, sc_h, 'bilinear')
       flow = image.scale(flow, sc_w, sc_h, 'bilinear')
       flow[2]:mul(sc_w / in_w); 
       flow[1]:mul(sc_h / in_h)
   end
     
   local images  
   if need_aug then -- Train Val / Data Augmentation
        -- Increase the intensity of pixel where total internal reflection happens
       local dark  = rho:lt(0.70)
       local dark3 = dark:view(1,sc_h,sc_w):expand(3,sc_h,sc_w)
       image_tar[dark3] = image_tar[dark3] + torch.uniform(0.01, 0.2)
        -- Get the regions of boundary and total internal reflection 
       local final
       local k_e = torch.random(1,1)*2+1
       local mask_roi = image.erode(mask[1], torch.ones(k_e, k_e):float()):float()
       mask_roi:add(mask, -1, mask_roi) --mask_roi = mask - mask_roi
       mask_roi:add(dark[{{1}}]:float()):clamp(0, 1)
        -- Blur the boudary of the objects to make it look more real
       if torch.uniform(0, 1) > 0.5 then -- dilate mask_roi
           local k_d = 3 
           mask_roi = image.dilate(mask_roi[1], torch.ones(k_d, k_d):float()):float()
       end
       if path_tar:find('_cplx') and torch.uniform(0,1) > 0.5 then 
           -- Use a different smoothing strategy for complex shape
           local gs_k=image.gaussian(torch.random(2)*2+1, torch.uniform(0.25,0.5), 1, true)
           local blur_img = image.convolve(image_tar, gs_k, 'same')
           local mask3 = mask:expand(3, sc_h, sc_w)
           final = torch.cmul(mask3, blur_img) + torch.cmul((1- mask3),image_tar)
       else
           local mask_roi3 = mask_roi:view(1,sc_h,sc_w):expand(3,sc_h,sc_w)
           local gs_k=image.gaussian(torch.random(4)*2+1, torch.uniform(0.25,0.5), 1, true)
           local blur_img = image.convolve(image_tar, gs_k, 'same')
           final = torch.cmul(mask_roi3, blur_img) + torch.cmul((1- mask_roi3),image_tar)
       end
       flow[3][dark] = 0
       if self.opt.in_bg then --Stereo version 
           image_ref = self:preprocess_single()(image_ref)
       end

       images = torch.cat(image_ref, final, 1)
       local noise = torch.rand(image_ref:size()):repeatTensor(2,1,1)
       images = images:add(noise:csub(0.5):mul(self.opt.noise):float())
       images = self:preprocess()(images):clamp(0, 1)
       mask_roi = mask_roi:byte()
       rho[mask_roi] = images:narrow(1,4,3):max(1)[mask_roi]
   else
       images = torch.cat(image_ref, image_tar, 1)
       flow[{{3}}][rho:le(0.7)] = 0 
   end

   if need_flip then
       if torch.uniform(0, 1) > 0.8 and not path_tar:find('water') then
           image.vflip(images, images)
           image.vflip(mask, mask)
           image.vflip(rho, rho)
           image.vflip(flow, flow); flow[1]:mul(-1) 
       else
           image.hflip(images, images)
           image.hflip(mask, mask)
           image.hflip(rho, rho)
           image.hflip(flow, flow); flow[2]:mul(-1) 
       end
   end
   if need_rotate then
       local ang = torch.uniform(0, 2) * self.opt.rot_ang - self.opt.rot_ang 
       images = image.rotate(images, ang)
       mask   = image.rotate(mask, ang)
       rho    = image.rotate(rho, ang)
       flow   = self:rotateFlow(flow, ang)
   end

   if need_crop then
       local h_1 = math.floor(torch.uniform(1e-2, sc_h - crop_h))
       local w_1 = math.floor(torch.uniform(1e-2, sc_w - crop_w))
       images = image.crop(images, w_1, h_1, w_1 + crop_w, h_1 + crop_h)
       mask   = image.crop(mask,   w_1, h_1, w_1 + crop_w, h_1 + crop_h)
       rho    = image.crop(rho,    w_1, h_1, w_1 + crop_w, h_1 + crop_h)
       flow   = image.crop(flow,   w_1, h_1, w_1 + crop_w, h_1 + crop_h)
   end

   local sample = {
      input = images,
      mask = mask+1,
      rho  = rho,
      flow = flow
   }

   if self.opt.in_trimap then
       local choice = torch.uniform(0, 1)
       if choice < 0.3 then
           sample.trimap = self:_generateTrimapMixed(mask, crop_h, crop_w)
       elseif choice < 0.7 then
           sample.trimap = self:_generateTrimapTransform(mask, crop_h, crop_w)
       else
           sample.trimap = self:_generateTrimapErosion(mask, crop_h, crop_w)
       end 
   end
   collectgarbage()
   return sample
end
function TOMDataset:_generateTrimapMixed(mask, crop_h, crop_w)
   local trimap
   local index = torch.nonzero(mask[1])
   if index:nDimension() == 0 then -- no object
       trimap = mask[1]:clone():fill(0)
   else
       local rows = index:narrow(2, 1, 1)
       local cols = index:narrow(2, 2, 1)
       local t, b, l, r = rows:min(), rows:max(), cols:min(), cols:max()

       local i = torch.random(1, #self.img_info)
       local path_tar  = self.img_info[i]
       local path_base = str_utils.splitext(path_tar)
       local path_mask = path_base .. '_mask.png'
       local mask2
       mask2 = self:_loadImage(paths.concat(self.dir, path_mask), 1):float():div(255)
       mask2 = mask2[1]
       local index2 = torch.nonzero(mask2)
       if index2:nDimension() == 0 then -- no object
           local trimap = mask[1]:clone():fill(0)
           return trimap
       end
       local rows2, cols2 = index2:narrow(2, 1, 1), index2:narrow(2, 2, 1)
       local t2, b2, l2, r2 = rows2:min(), rows2:max(), cols2:min(), cols2:max()
       mask2 = image.scale(mask2[{{t2, b2}, {l2, r2}}], r - l + 1, b - t + 1, 'simple')
       local fg = mask[1]:clone():fill(0)
       fg[{{t,b}, {l,r}}] = mask2
       local fg = (mask[1] + fg):gt(1.5):float()

       local k_e = torch.random(2, 30)
       fg = image.erode(fg, torch.ones(k_e, k_e):float()):float()

       local unknown = fg:clone():fill(0)
       unknown[{{t,b}, {l,r}}] = 1
       trimap = fg + unknown
   end
   return trimap
end

function TOMDataset:_generateTrimapTransform(mask, crop_h, crop_w)
   local trimap
   local index = torch.nonzero(mask[1])
   if index:nDimension() == 0 then -- no object
       trimap = mask[1]:clone():fill(0)
   else
       local rows = index:narrow(2, 1, 1)
       local cols = index:narrow(2, 2, 1)
       local t, b, l, r = rows:min(), rows:max(), cols:min(), cols:max()

       local ang = torch.uniform(0, 2) * 1.2 - 1.2 
       local m_transform = image.rotate(mask[1], ang)  -- translate, rotate
       local x, y = torch.random(0, 20), torch.random(0, 20)
       m_transform = image.translate(m_transform, x, y)
       local fg = (mask[1] + m_transform):gt(1.5):float()

       local k_e = torch.random(2, 30)
       fg = image.erode(fg, torch.ones(k_e, k_e):float()):float()

       local unknown = fg:clone():fill(0)
       unknown[{{t,b}, {l,r}}] = 1
       trimap = fg + unknown
   end
   return trimap
end

function TOMDataset:_generateTrimapErosion(mask, crop_h, crop_w)
   local trimap
   local index = torch.nonzero(mask[1])
   if index:nDimension() == 0 then -- no object
       trimap = mask[1]:clone():fill(0)
   else
       local rows = index:narrow(2, 1, 1)
       local cols = index:narrow(2, 2, 1)
       local t, b, l, r = rows:min(), rows:max(), cols:min(), cols:max()

       local k_e = torch.random(2, 30)
       local fg = image.erode(mask[1], torch.ones(k_e, k_e):float()):float()
       local unknown = fg:clone():fill(0)
       unknown[{{t,b}, {l,r}}] = 1
       trimap = fg + unknown
   end
   return trimap
end

function TOMDataset:_loadImage(path, channels)
   local ok, input = pcall(function()
       if channels == 1 then
           return image.load(path, channels, 'byte')
       else
           return image.load(path, channels, 'float')
       end
   end)
   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()
      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))
      input = image.decompress(b, 3, 'float')
   end
   return input 
end

function TOMDataset:size()
   return #self.img_info
end

function TOMDataset:preprocess_single()
   local randTransform = torch.rand(1)
   randTransform[1]= 1
   if randTransform[1] > 0.5 and (self.split == 'train' or self.split=='val')then 
      return t.Compose{
          t.ColorJitterSingle({
             brightness = 0.2,
             contrast = 0.2,
             saturation = 0.2,
          })
      }
  end
end

function TOMDataset:preprocess()
   local randTransform = torch.rand(1)
   randTransform[1]= 1
   if randTransform[1] > 0.5 and (self.split == 'train' or self.split=='val')then 
      return t.Compose{
          t.ColorJitter({
             brightness = 0.2,
             contrast = 0.2,
             saturation = 0.2,
          })
      }
  end
end

return M.TOMDataset
