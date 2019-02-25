require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

local io_utils   = require 'utils/io_utils'
local dict_utils = require 'utils/dict_utils'
local str_utils  = require 'utils/str_utils'
local flow_utils = require 'utils/flow_utils'
local eval_utils = require 'eval/eval_utils'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()

-- Model Options
cmd:option('-c_net', 'data/TOM-Net_model/CoarseNet.t7', 'CoarseNet')
cmd:option('-r_net', 'data/TOM-Net_model/RefineNet.t7', 'RefineNet') 

-- Input Options
cmd:option('-input_root', 'data/datasets/TOM-Net_Synth_Val_900/')
cmd:option('-in_bg',       false,               'Takes background image as input')
cmd:option('-img_dir',    'Images',     'synthetic data')
cmd:option('-img_list',   'glass.txt',  '')
cmd:option('-max_img_num',  -1,         'Larger than 0 to enable this option')
cmd:option('-w',           448,         'Image width') 
cmd:option('-h',           448,         'Image height')

local opt = cmd:parse(arg)

function saveResults(save_prefix, data, results)
    results.fcolor    = flow_utils.flowToColor(results.flow)
    results.mask_gt   = data.mask
    results.rho_gt    = data.rho
    results.fcolor_gt = flow_utils.flowToColor(data.flow)
    results.fcolor_gt[data.rho:float():lt(0.7):view(1, opt.h, opt.w):expand(3, opt.h, opt.w)] = 0
    io_utils.save_result_separate(save_prefix, results)
end

function getFinalPredImg(ref, pred_img, mask, rho)
    local rho = rho:repeatTensor(1, 3, 1, 1)
    local final_pred_img = torch.cmul(1 - mask, ref) + torch.cmul(mask, torch.cmul(pred_img, rho))
    return final_pred_img
end

function check_quant_res(data, pred)
    local errors = {}
    errors.I_MSE   = eval_utils.calMSE(data.tar, pred.in_rec)
    errors.I_MSE_B = eval_utils.calMSE(data.tar, data.bg)

    errors.M_IoU   = eval_utils.calIoUMask(data.mask:squeeze(), pred.mask:squeeze()[1])
    errors.M_IoU_B = eval_utils.calIoUMask(data.mask:squeeze(), false)
    errors.R_MSE   = eval_utils.calMSE(data.rho:squeeze(), pred.rho:squeeze())
    errors.R_MSE_B = eval_utils.calMSE(data.rho:squeeze(), false)

    local e, e_roi = flow_utils.cal_epe_flow(data.flow, pred.flow:squeeze(), data.rho:gt(0.7), data.mask:squeeze())
    errors.F_EPE   = e
    errors.F_RoI   = e_roi
    local bg_e, bg_e_roi = flow_utils.cal_epe_flow(data.flow, false, data.rho:gt(0.7), data.mask:squeeze())
    errors.F_EPE_B = bg_e
    errors.F_RoI_B = bg_e_roi
    return errors
end

function runImage(c_net, r_net, warp_module, img_path, idx)
    local param  = {img_path=img_path, img_dir=opt.img_dir, h=opt.h, w=opt.w, cuda=true,
                   in_bg = opt.in_bg}
    local data   = eval_utils.getInputData(param)
    local coarse = c_net:forward(data.input)
    coarse = coarse[#coarse] 
    local c_flow, c_mask, c_rho = coarse[1], coarse[2], coarse[3]
    local refine_in = {data.input, c_flow, c_mask, c_rho} -- flow, mask, rho
    output = r_net:forward(refine_in) 

    local r_flow, r_rho = output[1], output[2]
    local c_mask = eval_utils.getMask(c_mask, true)
    local bg_img = data.bg:view(1, 3, opt.h, opt.w)
    local pred_img = getFinalPredImg(bg_img, warp_module:forward({bg_img, r_flow}), c_mask, r_rho)

    local results  = {}
    results.flow   = r_flow:squeeze()
    results.mask   = c_mask:squeeze()
    results.rho    = r_rho:squeeze()
    results.input  = data.tar:squeeze()
    results.in_rec = pred_img:squeeze()
    results.bg     = data.bg:squeeze()

    local loss = check_quant_res(data, results)
    print_result(loss)

    local save_dir = paths.concat(opt.save_path, idx)
    paths.mkdir(save_dir)
    local prefix, _ = str_utils.splitext(img_path[1])
    local save_prefix = paths.concat(save_dir, string.format('%s',prefix))
    saveResults(save_prefix, data, results)
    return loss
end

function get_img_paths()
    local img_paths, num_img = io_utils.read_list(opt.img_list, true)
    num_img = opt.max_img_num > 0 and opt.max_img_num or #img_paths
    print(string.format('[Image number]: %d', #img_paths))
    return img_paths, num_img
end

function getModel(checkp_path)
    print(string.format('[Model]: Loading model from %s', checkp_path))
    local model = torch.load(checkp_path).model
    model:evaluate()
    return model
end

function setDefault()
    local root = paths.dirname(opt.r_net)
    local date, time = str_utils.getDateTime()
    local prefix  = string.format('%s', date) 
    local suffix  = paths.basename(opt.input_root) .. '_' .. str_utils.splitext(opt.img_list)
    opt.save_path = paths.concat(root, prefix .. '_synth_' .. suffix)

    opt.img_dir   = paths.concat(opt.input_root, opt.img_dir)
    opt.img_list  = paths.concat(opt.input_root, opt.img_list)
    print(string.format('[Paths]: Data Root: %s',  opt.input_root))
    print(string.format('[Paths]: Image Dir: %s',  opt.img_dir))
    print(string.format('[Paths]: Image List: %s', opt.img_list))
    paths.mkdir(opt.save_path)
    cmd:log(opt.save_path .. '/logfile', opt)
end

function print_result(errors)
    local item = {'F_EPE', 'F_RoI', 'M_IoU', 'R_MSE', 'I_MSE'}
    for i = 1, #item do
        local sub = {}
        for k, v in pairs(errors) do
            if k:find(item[i]) then sub[k] = v end
        end
        if dict_utils.dictLength(sub) ~= 0 then 
            print(string.format('\t[%s]:%s', item[i], str_utils.build_loss_string(sub, true))) 
        end
    end
end

setDefault()
local img_paths, num_img = get_img_paths()
local c_net = getModel(opt.c_net)
local r_net = getModel(opt.r_net)

local model_utils = require 'models/model_utils'
local warp_module = model_utils.createSingleWarpingModule():cuda()

local totalTimer = torch.Timer(); 
local times = {total = 0, iter = 0}


local summary = {}
for i = 1, num_img do
    local img_path = img_paths[i]
    print(string.format('[%d/%d] processing image %s', i, num_img, img_path[1]))
    local loss  = runImage(c_net, r_net, warp_module, img_path, i)
    summary[i]  = loss
    times.iter  = totalTimer:time().real
    times.total = str_utils.add_time(times.total, totalTimer)
    cutorch.synchronize()
    print(string.format('\t[Time Elapse: %.3fs]: Iter Time: %.3fs', times.total, times.iter))
end

local avg_summary = dict_utils.dictOfDictAverage(summary)
print("\n************[Finish Testing, Average]******************")
print_result(avg_summary)
torch.save(paths.concat(opt.save_path, 'summary.t7'),     summary)
torch.save(paths.concat(opt.save_path, 'avg_summary.t7'), avg_summary)
