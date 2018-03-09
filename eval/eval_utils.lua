require 'image'
require 'torch'

local eval_utils = {}
local dict_utils = require 'utils/dict_utils'
local io_utils   = require 'utils/io_utils'
local flow_utils = require 'utils/flow_utils'
local str_utils  = require 'utils/str_utils'

function eval_utils.getMask(masks, repeat3)
    local n, c, h, w = table.unpack(masks:size():totable())
    local m = masks:transpose(2,4):transpose(2,3)
    m = m:reshape(m:numel()/m:size(4), m:size(4))
    local _, pred = m:max(2)
    pred = pred:reshape(n, 1, h, w)
    pred:add(-1)
    if repeat3 then pred = pred:expand(n,3,h,w) end
    return pred:cuda()
end

function eval_utils.getFinalPred(ref_img, pred_img, pred_mask, pred_rho)
    local final_pred_img = torch.cmul(1 - pred_mask, ref_img) + torch.cmul(pred_mask, torch.cmul(pred_img, pred_rho))
    return final_pred_img
end

function eval_utils.getBgMaskRhoPath(img_path)
    local name, ext = str_utils.splitext(img_path)
    bg_path   = name .. '_ref.jpg'
    flow_path = name .. '_flow.flo'
    mask_path = name .. '_mask.png' 
    rho_path  = name .. '_rho.png'
    return bg_path, flow_path, mask_path, rho_path
end

function eval_utils.getInputData(param)
    local h = param.h; local w = param.w
    local bg_path, flow_path, mask_path, rho_path = eval_utils.getBgMaskRhoPath(param.img_path[1])
    local img_ref = image.load(paths.concat(param.img_dir, bg_path), 3)
    local img_tar = image.load(paths.concat(param.img_dir, param.img_path[1]), 3)
    local gt_mask = image.load(paths.concat(param.img_dir, mask_path))
    local gt_rho  = image.load(paths.concat(param.img_dir, rho_path))
    local gt_flow = flow_utils.loadShortFlowFile(paths.concat(param.img_dir, flow_path))

    local _, f_h, f_w = table.unpack(gt_flow:size():totable())
    if h ~= f_h or w ~=f_w then
        img_ref = image.scale(img_ref, w, h, 'bilinear')
        img_tar = image.scale(img_tar, w, h, 'bilinear')
        gt_mask = image.scale(gt_mask, w, h, 'simple')
        gt_rho  = image.scale(gt_rho,  w, h, 'simple')
        gt_flow = image.scale(gt_flow, w, h, 'bilinear')
        gt_flow[2]:mul(param.w / f_w); 
        gt_flow[1]:mul(param.h / f_h)
    end
    img_ref, img_tar, rho = eval_utils.preprocess(param, img_ref, img_tar, gt_mask, gt_rho)
    local input = img_tar:view(1, 3, h, w)
    if param.cuda then
        input   = input:cuda()
        img_ref = img_ref:cuda()
        gt_flow = gt_flow:cuda()
        gt_mask = gt_mask:cuda()
        gt_rho  = gt_rho:cuda()
    end
    local data = {input = input, bg = img_ref, flow=gt_flow, mask=gt_mask, rho=gt_rho}
    return data
end

function eval_utils.preprocess(param, img_ref, img_tar, mask, rho)
    -- Increase the intensity of dark region
    local dark     = rho:lt(0.70)
    local dark3    = dark:view(1, param.h, param.w):expand(3, param.h, param.w)
    local offset   = 0.05
    img_tar[dark3] = img_tar[dark3] + offset

    -- Get Boundary
    local mask_roi  = image.erode(mask[1], torch.ones(3, 3):float()):float()
    mask_roi:add(mask, -1, mask_roi)
    mask_roi:add(dark[{{1}}]:float()):clamp(0, 1)
    local mask_roi3 = mask_roi:view(1, param.h, param.w):expand(3, param.h, param.w)
    -- Blur image
    local k_sz     = 5 
    local gs_k     = image.gaussian(5, 0.3, 1, true)
    local blur_img = image.convolve(img_tar, gs_k, 'same')
    local final   = torch.cmul(mask_roi3, blur_img) + torch.cmul((1-mask_roi3), img_tar)
    param.convertToCuda = true
    rho[mask_roi:byte()] = final:max(1)[mask_roi:byte()]
    return img_ref, final, rho
end

function eval_utils.getModel(checkp_path, no_warping, use_cudnn)
    if not paths.filep(checkp_path) then
       local name, ext = str_utils.splitext(checkp_path)
       local f = io.open(paths.concat(paths.dirname(checkp_path), 'latest'), 'r') 
       local suffix = f:read()
       f:close()
       checkp_path = name .. suffix .. ext
    end
    print(string.format('[Model]: Loading model from %s', checkp_path))

    local model = torch.load(checkp_path).model:cuda()

    if ues_cudnn then
        require 'cudnn'
        cudnn.fastest = true
        cudnn.benchmark = true
        cudnn.convert(model, cudnn)
    end
    model:evaluate()
    if no_warping then
        return model
    end
    print('[Model]: Load warping module')
    require 'models/flow_warping_module'
    local warping_module  = createSingleWarpingModule():cuda()
    return model, warping_module
end

function eval_utils.calErrRho(gt_rho, pred_rho, roi, gt_mask)
    local roi = (roi == nil) and true or roi
    local error_rho
    if pred_rho ~= false then
        local pred_rho = pred_rho:squeeze()
        pred_rho = (pred_rho:size(1) == 3) and pred_rho:select(1,1) or pred_rho
        error_rho = torch.csub(gt_rho:squeeze(), pred_rho):abs():sum()
    else
        error_rho = torch.csub(gt_rho:squeeze(), 1):abs():sum()
    end

    if roi then 
        local roi_ratio = (gt_mask:ge(0.5):sum()) / gt_mask:nElement()
        if roi_ratio == 0 then roi_ratio = 1 end
        error_rho = error_rho / gt_rho:nElement() / roi_ratio
    else
        error_rho = error_rho / gt_rho:nElement()
    end
    return error_rho * 100
end

function eval_utils.calIoUMask(gt_mask, pred_mask)
    local t_p, f_pn
    if pred_mask ~= false then
        local pred_mask = pred_mask:squeeze()
        pred_mask = (pred_mask:size(1) == 3) and pred_mask:select(1,1) or pred_mask
        t_p = (gt_mask + pred_mask):gt(1.5):sum()
        f_pn = (gt_mask + pred_mask):eq(1):sum()
    else
        t_p = gt_mask:gt(0.5):sum()
        f_pn = gt_mask:lt(0.1):sum()
    end

    local IoU 
    if (t_p == 0) and (f_pn == 0) then
        IoU = 1
    else
        IoU = t_p / (t_p + f_pn)
    end
    return IoU
end

function eval_utils.calMSE(gt, pred)
    local diff
    if pred == false then
        diff = (gt-1):pow(2):mean()
    else
        diff = torch.csub(gt, pred):pow(2):mean()
    end
    return diff * 100
end

return eval_utils
