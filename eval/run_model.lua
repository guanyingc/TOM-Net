require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

local io_utils   = require 'utils/io_utils'
local str_utils  = require 'utils/str_utils'
local flow_utils = require 'utils/flow_utils'
local eval_utils = require 'eval/eval_utils'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()

-- Model Options
cmd:option('-c_net', 'data/TOM-Net_model/CoarseNet.t7', 'CoarseNet')
cmd:option('-r_net', 'data/TOM-Net_model/RefineNet.t7', 'RefineNet') 

-- Input Options
cmd:option('-input_img',   '',      'If empty, use input_root and img_list to specify input')
cmd:option('-input_root',  'data/datasets/TOM-Net_Real_Test_876')
cmd:option('-img_list',    'Sample_paper.txt',  'Name of image list')
cmd:option('-have_ref',    true,                'Input have background image')
cmd:option('-in_trimap',   false,               'Takes trimap as input')
cmd:option('-in_bg',       false,               'Takes background image as input')
cmd:option('-max_img_num',  -1,                 'Maximum test number of images')
cmd:option('-width',       448,                 'Image width')
cmd:option('-height',      448,                 'Image height')

local opt = cmd:parse(arg)

function getInputData(img_path)
   local w, h    = opt.width, opt.height
   local img_tar = image.load(paths.concat(opt.input_root, img_path[1]), 3)
   local input
   img_tar = image.scale(img_tar, w, h, 'bilinear'):cuda()
   local img_ref
   if opt.have_ref then
       img_ref = image.load(paths.concat(opt.input_root, img_path[2]), 3)
       img_ref = image.scale(img_ref, w, h, 'bilinear'):cuda()
   end
   --img_tar = img_tar:view(1, 3, h, w)
   local input, trimap
   if opt.in_bg then
       input = img_tar.new():resize(1, 6, h, w)
       input[{{1}, {1,3}}] = img_ref
       input[{{1}, {4,6}}] = img_tar
   elseif opt.in_trimap then
       trimap = image.load(paths.concat(opt.input_root, img_path[3]))
       if trimap:nDimension() == 3 then
           trimap = trimap[{1}]
       end
       input = img_tar:view(1, 3, h, w)
       local fg, bg = trimap:gt(0.7), trimap:lt(0.3) 
       trimap:fill(1)
       trimap[fg] = 2
       trimap[bg] = 0
       trimap = image.scale(trimap, w, h, 'simple')
       input = img_tar.new():resize(1, 4, h, w)
       input[{{1}, {1,3}}] = img_tar
       input[{{1}, {4}}] = trimap
   else
       input = img_tar:view(1, 3, h, w)
   end
   local data = {input = input, tar = img_tar, bg = img_ref}
   if opt.in_trimap then
       data.trimap = trimap
   end
   return data
end

function runImage(c_net, r_net, img_path, idx)
    local data   = getInputData(img_path)
    local coarse = c_net:forward(data.input)
    coarse = coarse[#coarse] 
    local c_flow = coarse[1]
    local c_mask = coarse[2]
    local c_rho  = coarse[3]
    local refine_in = {data.input, c_flow, c_mask, c_rho} -- flow, mask, rho
    output = r_net:forward(refine_in) 

    local results = {}
    results.flow   = output[1]:float():squeeze()
    results.fcolor = flow_utils.flowToColor(results.flow)
    results.mask   = eval_utils.getMask(c_mask):float():squeeze()
    results.rho    = output[2]:float():squeeze()
    results.input  = data.tar:float():squeeze()
    if data.bg then -- If have background image
        results.bg = data.bg:float():squeeze()
    end
    if data.trimap then
        results.trimap = data.trimap:float():squeeze() / 2.0
    end

    local img_dir  = string.format('%d_%s', idx, str_utils.splitext(paths.basename(img_path[1])))
    local save_dir = paths.concat(opt.save_path, img_dir)
    paths.mkdir(save_dir)
    io_utils.save_result_separate(paths.concat(save_dir, img_dir), results)
    return img_dir
end

function getImagePaths()
    local img_paths, num_img
    if opt.input_img ~= '' then
        opt.input_root = paths.dirname(opt.input_img)
        img_paths      = { {paths.basename(opt.input_img)} }
    else
        img_paths, num_img = io_utils.read_list(opt.img_list, true)
    end
    num_img = opt.max_img_num > 0 and opt.max_img_num or #img_paths
    print(string.format('[Image number]: %d', #img_paths))
    return img_paths, num_img
end

function getModel(checkp_path)
    print(string.format('\n[Model]: Loading model from %s', checkp_path))
    local model = torch.load(checkp_path).model
    model:evaluate()
    return model
end

function setDefault()
    local prefix  = string.format('%s', str_utils.getDateTime())
    local suffix
    if opt.input_img == '' then
        suffix = paths.basename(opt.input_root) .. '_' .. str_utils.splitext(opt.img_list)
        opt.img_list = paths.concat(opt.input_root, opt.img_list)
        print(string.format('[Paths]: Data Root: %s',  opt.input_root))
        print(string.format('[Paths]: Image List: %s', opt.img_list))
    else
        print(string.format('[Paths]: Testing Image: %s', opt.input_img))
        suffix = paths.basename(opt.input_img)
        opt.have_ref = false
    end
    opt.save_path = paths.concat(paths.dirname(opt.r_net), prefix .. '_real_' .. suffix)
    paths.mkdir(opt.save_path)
    cmd:log(opt.save_path .. '/logfile', opt) -- Logger
end

setDefault()
local img_paths, num_img = getImagePaths()
local c_net = getModel(opt.c_net)
local r_net = getModel(opt.r_net)

local totalTimer = torch.Timer(); 
local times = {total = 0, iter = 0}
local dir_list = {}

for i = 1, num_img do
    local img_path = img_paths[i]
    print(string.format('[%d/%d] processing image %s', i, num_img, img_path[1]))
    local img_dir = runImage(c_net, r_net, img_path, i)
    dir_list[i]   = img_dir
    times.iter    = totalTimer:time().real
    times.total   = str_utils.add_time(times.total, totalTimer)
    cutorch.synchronize()
    print(string.format('\t [Time Elapse: %.3fs]: Iter Time: %.3fs', times.total, times.iter))
end
io_utils.save_list(paths.concat(opt.save_path, 'dir_list.txt'), dir_list)

if opt.have_ref then
    print('****** Evaluating for real data ******')
    cmd = string.format('python eval/eval_real_data.py --input_root %s --dir_list %s', opt.save_path, 'dir_list.txt')
    os.execute(cmd)
end
