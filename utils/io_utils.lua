require 'torch'
require 'math'
require 'paths'
require 'gnuplot'
require 'image'
local dict_utils = require 'utils/dict_utils'
local flow_utils = require 'utils/flow_utils'
local io_utils = {}

-- Read and Write file 
function io_utils.loadt7(condition, fname)
    local t7_file = {}
    if condition then
        if paths.filep(fname) then
            t7_file = torch.load(fname)
        end
    end
    return t7_file
end

function io_utils.save_list(save_name, list)
    fout = io.open(save_name, 'w')
    for k, fname in pairs(list) do
        fout:write(string.format('%s\n', fname))
    end
    fout:close()
end

function io_utils.value_in_table(value, table)
    local is_in = false
    for k, v in pairs(table) do
        if value == v then is_in = true end
    end
    return is_in
end

function io_utils.get_img_list(img_dir, split, exts)
    local exts = exts or {'jpg', 'png', 'JPG'}
    local list = paths.dir(img_dir)
    table.sort(list)
    local img_list = {}
    for k, fname in pairs(list) do
        if io_utils.value_in_table(paths.extname(fname), exts) then
            if split then fname = string.split(fname, ' ' ) end
            table.insert(img_list, fname)
        end
    end
    return img_list
end

function io_utils.read_list(img_list, split, max_image_num)
   print('Get input images')
   local image_paths = {}
   local counter = 0
   if img_list ~= '' then
      for names in io.lines(img_list) do
         counter = counter + 1
         image_paths[counter] = split and string.split(names, ' ') or names
      end
   else
      error('one of the img_list must be provdied.', img_list)
   end
   local max_image_num = max_image_num or -1
   local num_images = max_image_num > 0 
         and math.min(#image_paths, max_image_num) or #image_paths
   print('num_images', num_images)
   return image_paths, num_images
end

-- Save Visualization Results
function io_utils.plot_results_compact(results, logDir, split)
   print(string.format('Ploting compact figures for %s', split))
    local items = {'_img', '_flow', '_smo', '_rho', '_mask', '_bg', 'flow_epe', 'rho_error', 'mask_error', '_grad'}
    for i = 1, #items do
        local item = items[i]
        local lines = {}
        for k, v in pairs(results) do
            if k:find(item) then
                table.insert(lines, {k, torch.range(1, #v), torch.Tensor(v), '-'})
            end
        end
        if #lines ~= 0 then
           f = gnuplot.pngfigure(paths.concat(logDir, split, item .. '.png'))
           gnuplot.plot(unpack(lines))
           gnuplot.plotflush()
           gnuplot.close(f)
        end
    end
end

function io_utils.save_results_compact(save_name, results, width_num)
    local int = 5
    local num = dict_utils.dictLength(results); 
    local w_n  = width_num or 3 ; local h_n = math.ceil(num / w_n)
    local big_Img
    local idx = 1
    local c, h, w
    local fix_h, fix_w
    for k, v in pairs(results) do
        if v ~= false then
          local img = v:float()
          if img:dim() > 3 or img:dim() <2 then error('Dim of image must be 2 or 3') end
          if not big_Img then
              c, h, w = table.unpack(img:size():totable())
              fix_h = h
              fix_w = w
              big_Img = torch.Tensor(3, h_n*h + (h_n-1)*int, w_n*w + (w_n-1)*int):fill(0)
          end
          if img:size(1) ~= 3 then img = torch.repeatTensor(img,3,1,1) end
          if img:size(2) ~= fix_h or img:size(3) ~=fix_w then
             img = image.scale(img, fix_w, fix_h, 'simple')
          end
          local h_idx= math.floor((idx-1)/w_n)+1; local w_idx=(idx-1)%w_n+1
          local h_start = 1 + (h_idx-1)*(h+int); 
          local w_start = 1 + (w_idx-1)*(w+int);
          big_Img[{{},{h_start,h_start+h-1},{w_start,w_start+w-1}}] = img
        end
        idx = idx + 1
    end
    image.save(save_name, big_Img)
end

function io_utils.save_result_separate(prefix, results)
    for k,v in pairs(results) do 
        local save_name = prefix .. '_' .. k
        if k == 'mask' or k == 'rho' then
            save_name = save_name .. '.png'
        elseif k == 'flow' then
            save_name = prefix .. '.flo'
        else
            save_name = save_name .. '.jpg'
        end
        if k == 'flow' then
            flow_utils.writeShortFlowFile(save_name, v)
        else
            image.save(save_name, v)
        end
    end
end
return io_utils
