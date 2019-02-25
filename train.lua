local optim       = require 'optim'
local model_utils = require 'models/model_utils'
local dict_utils  = require 'utils/dict_utils'
local str_utils   = require 'utils/str_utils'
local io_utils    = require 'utils/io_utils'
local flow_utils  = require 'utils/flow_utils'
local eval_utils  = require 'eval/eval_utils'
local M = {}
local Trainer = torch.class('TOM.Trainer', M)

function Trainer:setupMultiScaleData(opt)
    print('[Multi-Scale] Setting up MultiScale Data')
    -- Generate multi-scale ground truth during training
    local multiScaleData = model_utils.createMultiScaleData(opt):cuda()
    return multiScaleData
end

function Trainer:setupWarping(opt)
    print('Setting up warping module')
    local warping_module 
    if opt.refine then
        print('[Single Scale] Setting up SingleScaleWarping')
        warping_module = model_utils.createSingleWarpingModule():cuda()
        self.c_warp = model_utils.createSingleWarpingModule():cuda() -- For CoarseNet
    else
        print('[Multi-Scale] Setting up MultiScaleWarping')
        warping_module = model_utils.createMultiScaleWarping(opt.ms_num):cuda()
    end
    return warping_module
end

function Trainer:setupCriterion(opt)
    print('Setting up Criterion')
    print('[Flow Loss] Setting up criterion for flow')
    require 'criterion/TOMCriterionFlow'
    self.flow_crit = nn.TOMCriterionFlow(opt)
    self.flow_crit:cuda()
    if opt.refine then -- FOR Refinement
        -- In refinement stage, an addition flow criterion is initialized 
        -- to calculate the EPE error for CoarseNet
        self.c_flow_crit = nn.TOMCriterionFlow(opt)
        self.c_flow_crit:cuda()
    end

    print('[Unsup Loss] Setting up criterion for mask, rho and reconstruction image')
    require 'criterion/TOMCriterionUnsup'
    -- Criterion for Mask, attenuation mask and reconstruction loss
    self.unsup_crit = nn.TOMCriterionUnsup(opt)
    self.unsup_crit:cuda()
end

function Trainer:setup_sovler(opt, inOptimState)
    local optimState, optimMethod
    if opt.solver == 'ADAM' then
       print('[Solver] Using ADAM solver')
       optimState           = inOptimState or {
          learningRate      = opt.lr,
          beta1             = opt.beta_1,
          beta2             = opt.beta_2,
       }
       optimMethod = optim.adam
    else
        error('Unknown Optimization method')
    end
    return optimState, optimMethod
end

function Trainer:__init(model, opt, optimState)
    print('Initializing Trainer')
    self.opt             = opt
    self.model           = model
    self.warping_module  = self:setupWarping(opt) -- Reconstruct input based on refractive flow field
    self.optimState, self.optimMethod = self:setup_sovler(opt, optimState) -- check if resume training
    self:setupCriterion(opt)
    if not opt.refine then
        -- In coarse stage, multi-scale ground truth matte is needed
        self.multiScaleData = self:setupMultiScaleData(opt)
    end

    print('Get model parameters and gradient parameters')
    self.params, self.gradParams = model:getParameters()
    print('Total number of parameters in TOM-Net: ', self.params:nElement())
    print('Total number of parameters gradient in TOM-Net: ', self.gradParams:nElement())
    -- Variable to store error for the estimated environment matte
    self.flow_e = 0; self.mask_e = 0; self.rho_e = 0
end

function Trainer:getRefineInput(input, predictor)
    local c_ls = {} -- loss in coarse stage
    local coarse = predictor:forward(input) 
    coarse = coarse[#coarse]

    c_ls.c_loss_flow  = self.c_flow_crit:forward(coarse[1], self.flows).loss_flow
    c_ls.flow_epe_c   = self:get_flow_error(self.c_flow_crit.epe)
    c_ls.mask_error_c = self:get_mask_error(coarse, true)
    c_ls.rho_error_c  = self:get_rho_error(coarse, true)

    local refine_in = {input, coarse[1], coarse[2], coarse[3]}
    return refine_in, coarse, c_ls
end

function Trainer:train(epoch, dataloader, split, predictor)
    local split = split or 'train'
    self.optimState.learningRate = self:learningRate(epoch)
    print(string.format('Epoch %d, Learning rate %.8f', epoch, self.optimState.learningRate) )
    local num_batches = dataloader:batch_size()
    print('=============================')
    print(self.optimState)
    print('=============================')
    print(string.format('=> Training epoch # %d, totaling mini batches %d', epoch, num_batches))

    self.model:training()
    local crit_output = 0.0
    local timer       = torch.Timer()
    local times       = {dataTime=0, modelTime=0, lossTime=0} 
    local loss        = {} -- loss every 20 iteration
    local losses      = {} -- loss whole epoch
    local num_batches = dataloader:batch_size()

    local function feval()
        return crit_output, self.gradParams
    end

    for iter, sample in dataloader:run(split, self.opt.max_image_num) do 
        local input = self:copyInputData(sample)
        times.dataTime = str_utils.add_time(times.dataTime, timer)

        local coarse, c_ls
        if self.opt.refine then
            input, coarse, c_ls = self:getRefineInput(input, predictor)
            dict_utils.dictsAdd(loss, c_ls)
        end

        local output = self.model:forward(input)
 
        local flows, pred_imgs = self:flowWarpingForward(output) -- warp input image with flow
        times.modelTime = str_utils.add_time(times.modelTime, timer)

        local unsup_loss, unsup_grads = self:unsupCritForwardBackward(output, pred_imgs)
        -- Loss and grads for object mask, attenuation mask and reconstruction loss
        dict_utils.dictsAdd(loss, unsup_loss)
        local warping_grads = self:flowWarpingBack(flows, unsup_grads)

        -- Loss and grads for refractive flow field (supervised loss)
        local sup_loss, sup_grads = self:supCritForwardBackward(flows)
        dict_utils.dictsAdd(loss, sup_loss)
        times.lossTime = str_utils.add_time(times.lossTime, timer)
                            
        -- Combine all the gradients for the network
        local model_grads = self:getModelGrads(unsup_grads, sup_grads, warping_grads)
        self.model:zeroGradParameters()
        self.model:backward(input, model_grads)
        
        -- Update parameters 
        local _, tmp_loss = optim.adam(feval, self.params, self.optimState)
        times.modelTime = str_utils.add_time(times.modelTime, timer)

        if (iter % self.opt.train_display) == 0 then
            losses[iter] = self:display(epoch, iter, num_batches, loss, times, split)
            dict_utils.dictReset(loss); dict_utils.dictReset(times)
        end

        if (iter % self.opt.train_save) == 0 then
            if self.opt.refine then
                self:saveRefineResults(epoch, iter, output, pred_imgs,split, 1, coarse)
            else
                self:saveMultiResults(epoch, iter, output, pred_imgs, split)
            end
            print(string.format('\t Save results time: %.4f', timer:time().real))
        end
        collectgarbage()
        assert(self.params:storage() == self.model:parameters()[1]:storage())
        timer:reset()
    end

    collectgarbage()
    local average_loss = dict_utils.dictOfDictAverage(losses)
    print(string.format(' | Epoch: [%d] Losses summary: %s', epoch, str_utils.build_loss_string(average_loss)))
    return average_loss
end

function Trainer:saveRefineResults(epoch, iter, output, pred_imgs, split, num, coarse)
    local split = split or 'train'
    local num = (num > 0 and num < output[1]:size(1)) and num or output[1]:size(1)
    local c_pred = self.c_warp:forward({self.ref_imgs, coarse[1]}) 
    for id = 1, num do 
        local gt_fcolor = flow_utils.flowToColor(self.flows[id])
        local results = {self.ref_imgs[id], self.tar_imgs[id], gt_fcolor, self.masks[id]-1, self.rhos[id]}
        local c_fcolor = flow_utils.flowToColor(coarse[1][id])
        local c_mask   = eval_utils.getMask(coarse[2][{{id}}], true):squeeze()
        local c_rho    = coarse[3][id]:repeatTensor(3, 1, 1)
        local coarse = {false, c_pred[id], c_fcolor, c_mask, c_rho}

        local r_fcolor = flow_utils.flowToColor(output[1][id])
        local r_rho    = output[2][id]:repeatTensor(3, 1, 1)
        local refine = {false, pred_imgs[id], r_fcolor, false, r_rho}

        for k, v in pairs(coarse) do table.insert(results, v) end
        for k, v in pairs(refine) do table.insert(results, v) end

        local save_name = self:getSaveName(self.opt.logDir, split, epoch, iter, id)
        io_utils.save_results_compact(save_name, results, 5)
    end
end
function Trainer:getSaveName(logDir, split, epoch, iter, id)
    local fPath  = string.format('%s/%s/Images/', logDir, split)
    local fNames = string.format('%s_%s_%s', epoch, iter, id)
    fNames = string.format('%s_EPE_%.2f_IoU_%.3f_Rho_%.1f', fNames, self.flow_e, self.mask_e, self.rho_e) 
    return paths.concat(fPath, fNames .. '.jpg')
end

function Trainer:getPredicts(split, id, output, pred_img, m_scale)
    local flow, color_flow, mask, rho, final_img 
    local pred = {}

    local gt_color_flow 
    if m_scale then
        gt_color_flow = flow_utils.flowToColor(self.multi_flows[m_scale][id])
    else
        gt_color_flow = flow_utils.flowToColor(self.flows[id])
    end
    table.insert(pred, gt_color_flow)

    local color_flow = flow_utils.flowToColor(output[1][id])
    table.insert(pred, color_flow)
    mask = eval_utils.getMask(output[2][{{id}}], true):squeeze()
    table.insert(pred, mask)
    rho = output[3][id]:repeatTensor(3,1,1)
    table.insert(pred, rho)

    local first_img
    if m_scale then
        final_img = eval_utils.getFinalPred(self.multi_ref_imgs[m_scale][id], pred_img[id], mask, rho)
        first_img = self.multi_tar_imgs[m_scale][id]
    else
        final_img = eval_utils.getFinalPred(self.ref_imgs[id], pred_img[id], mask, rho)
        first_img = self.tar_imgs[id]
    end
    table.insert(pred, 1, first_img)
    table.insert(pred, 2, final_img)
    return pred, final_img
end

function Trainer:getFirstRow(split, id)
    local first = {}
    table.insert(first, self.ref_imgs[id])
    table.insert(first, self.tar_imgs[id])
    table.insert(first, false)
    if self.opt.in_trimap then
        table.insert(first, self.trimaps[id] / 2.0)
    else
        table.insert(first, false)
    end
    table.insert(first, self.masks[id] - 1) 
    table.insert(first, self.rhos[id])
    return first
end
function Trainer:saveMultiResults(epoch, iter, output, multi_pred_img, split, id)
    local id = id or 1
    local scales = self.opt.ms_num
    local results = {}

    local first_row = self:getFirstRow(split, id)
    for k, v in pairs(first_row)  do table.insert(results, v) end

    for i = scales, 1, -1 do
        local pred_img = multi_pred_img[i]
        local sub_pred = self:getPredicts(split, id, output[i], pred_img, i)
        for k, v in pairs(sub_pred) do table.insert(results, v) end
    end

    local save_name = self:getSaveName(self.opt.logDir, split, epoch, iter, id)
    io_utils.save_results_compact(save_name, results, 6)
    print(string.format('\t Flow Magnitude: Max %.3f, Min %.3f, Mean %.3f', output[scales][1][id]:max(), output[scales][1][id]:min(), torch.abs(output[scales][1][id]):mean()))
    collectgarbage()
end
-------- FOR IMAGE RECONSTRUCTION LOSS AND IMAGE WARPING ----------
function Trainer:flowWarpingForward(output)
    local flows = {}
    local pred_imgs
    if self.opt.refine then
        flows = output[1]
        pred_imgs = self.warping_module:forward({self.ref_imgs, flows})
    else
        for i = 1, self.opt.ms_num do
            flows[i] = output[i][1]
        end
        pred_imgs = self.warping_module:forward({self.multi_ref_imgs, flows})
    end
    return flows, pred_imgs
end

function Trainer:flowWarpingBack(flows, unsup_grads)
    local crit_img_grads = {}
    local warping_grads
    if not self.opt.refine then --refine stage does not use rec_loss
        for i = 1, self.opt.ms_num do
            crit_img_grads[i] = unsup_grads[i][1]
        end
        warping_grads = self.warping_module:backward({self.multi_ref_imgs, flows}, crit_img_grads)[2]
    end
    return warping_grads
end

--------- For Erro Calculation -------------
function Trainer:get_mask_error(output, isCoarse)
    local gt_mask = self.masks[1] - 1 
    local mask = isCoarse and output[2][{{1}}] or output[#output][2][{{1}}]
    local pred_mask  = eval_utils.getMask(mask, false)
    self.mask_e = eval_utils.calIoUMask(gt_mask, pred_mask)
    return self.mask_e
end

function Trainer:get_rho_error(output, isCoarse)
    local gt_mask = self.masks[1] - 1
    local gt_rho  = self.rhos[1]
    local idx = (isCoarse or not self.opt.refine) and 3 or 2
    local rho = (isCoarse or self.opt.refine) and output[idx][1] or output[#output][idx][1]
    self.rho_e  = eval_utils.calErrRho(gt_rho, rho, true, gt_mask)
    return self.rho_e
end

function Trainer:get_flow_error(avg_epe)
    local roi_ratio = (self.masks-1):gt(0.5):sum() / self.masks:nElement()
    if roi_ratio == 0 then roi_ratio = 1 end
    self.flow_e = avg_epe / roi_ratio
    return self.flow_e
end

function Trainer:unsupCritForwardBackward(output, pred_imgs, forwardOnly)
    local crit_input = {}
    local crit_target = {}
    if self.opt.refine then
        table.insert(crit_input, output[2])
        table.insert(crit_target, self.rhos)
    else
        for i = 1, self.opt.ms_num do
            crit_input[i]  = {}
            crit_target[i] = {}
            local w_m = torch.cmul(self.multi_flows[i]:narrow(2,3,1), self.multi_rhos[i]):expandAs(self.multi_flows[i])
            table.insert(crit_input[i],  torch.cmul(pred_imgs[i], w_m))
            table.insert(crit_target[i], torch.cmul(self.multi_tar_imgs[i], w_m))

            table.insert(crit_input[i],  output[i][2])
            table.insert(crit_target[i], self.multi_masks[i])

            table.insert(crit_input[i],  output[i][3])
            table.insert(crit_target[i], self.multi_rhos[i])
        end
    end
    local ls_iter  = {} -- Loss in this iteration
    ls_iter.rho_error  = self:get_rho_error(output)

    if not self.opt.refine then
        ls_iter = self.unsup_crit:forward(crit_input, crit_target)
        ls_iter.mask_error = self:get_mask_error(output)
    end

    if forwardOnly then
        return ls_iter
    end
    local crit_grads = self.unsup_crit:backward(crit_input, crit_target)
    return ls_iter, crit_grads
end

function Trainer:supCritForwardBackward(flows, forwardOnly)
    local flow_crit_target = self.opt.refine and self.flows or self.multi_flows 

    local ls_iter    = self.flow_crit:forward(flows, flow_crit_target)
    ls_iter.flow_epe = self:get_flow_error(self.flow_crit.epe)

    if forwardOnly then
        return ls_iter
    end
    local flow_grads = self.flow_crit:backward(flows, flow_crit_target)

    return ls_iter, flow_grads
end

function Trainer:getModelGrads(unsup_grads, sup_grads, warping_grads)
    local model_grads = {}
    if self.opt.refine then
        local flow_grads  = sup_grads
        table.insert(model_grads, flow_grads)     -- flow
        table.insert(model_grads, unsup_grads[1]) -- rho
    else
        for i = 1, self.opt.ms_num do
            local flow_grads
            flow_grads = warping_grads[i]
            flow_grads:add(sup_grads[i]) 
            model_grads[i] = {flow_grads}               -- flow
            local unsup_grad = unsup_grads[i]
            table.insert(model_grads[i], unsup_grad[2]) -- mask
            table.insert(model_grads[i], unsup_grad[3]) -- rho
        end
    end
    return model_grads
end

function Trainer:test(epoch, dataloader, split, predictor)
    local timer = torch.Timer()
    local num_batches = dataloader:batch_size()

    local times  = {dataTime=0, modelTime=0, lossTime=0}
    local loss   = {}
    local losses = {} -- loss in whole epoch
    print(string.format('*** Testing after %d epoches ***', epoch))
    self.model:evaluate()

    for iter, sample in dataloader:run(split) do
        local input = self:copyInputData(sample)
        times.dataTime = str_utils.add_time(times.dataTime, timer)

        local coarse, c_ls
        if self.opt.refine then
            input, coarse, c_ls = self:getRefineInput(input, predictor)
            dict_utils.dictsAdd(loss, c_ls)
        end

        local output = self.model:forward(input)

        local flows, pred_imgs 
        flows, pred_imgs = self:flowWarpingForward(output)
        times.modelTime = str_utils.add_time(times.modelTime, timer, true)

        local unsup_loss, unsup_grads
        unsup_loss = self:unsupCritForwardBackward(output, pred_imgs, true)
        dict_utils.dictsAdd(loss, unsup_loss)

        local sup_loss = self:supCritForwardBackward(flows, true) 
        dict_utils.dictsAdd(loss, sup_loss)
        times.lossTime = str_utils.add_time(times.lossTime, timer)

        local val_disp = (split == 'val') and (iter % self.opt.val_display)  == 0
        if val_disp then
            losses[iter] = self:display(epoch, iter, num_batches, loss, times, split)
            dict_utils.dictReset(loss); dict_utils.dictReset(times)
        end

        local val_save   = (split == 'val')  and (iter % self.opt.val_save)  == 0

        if self.opt.refine then
            self:saveRefineResults(epoch, iter, output, pred_imgs, split, -1,coarse)
        elseif val_save then
            self:saveMultiResults(epoch, iter, output, pred_imgs, split)
        end

        collectgarbage()
        assert(self.params:storage() == self.model:parameters()[1]:storage())
        timer:reset()
    end
    local average_loss = dict_utils.dictOfDictAverage(losses)
    print(string.format(' | Epoch: [%d] Losses summary: %s', epoch, str_utils.build_loss_string(average_loss)))
    return average_loss
end

function Trainer:display(epoch, iter, num_batches, loss, times, split)
    local time_elapsed = str_utils.time_left(self.opt.startTime, self.opt.nEpochs, num_batches, epoch, iter)
    local interval = (split == 'train') and self.opt.train_display or self.opt.val_display
    loss_average = dict_utils.dictDivide(loss, interval)
    print(string.format(' | Epoch (%s): [%d][%d/%d] | %s', split, epoch, iter, num_batches, time_elapsed))
    print(str_utils.build_loss_string(loss_average))
    print(str_utils.build_time_string(times))
    return loss_average
end

function Trainer:copyInputData(sample)
    self:copyInputs(sample)
    if not self.opt.refine then
        self:copyInputsMultiScale(sample)
    end
    if self.opt.in_trimap then
        network_input = torch.cat(self.tar_imgs, self.trimaps, 2)
    elseif self.opt.in_bg then 
        network_input = torch.cat(self.ref_imgs, self.tar_imgs, 2)
    else
        network_input = self.tar_imgs
    end
    --local network_input = self.tar_imgs
    return network_input
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.ref_imgs   = self.ref_imgs or torch.CudaTensor()
   self.tar_imgs   = self.tar_imgs or torch.CudaTensor()
   self.masks      = self.masks    or torch.CudaTensor()
   self.rhos       = self.rhos     or torch.CudaTensor()
   self.flows      = self.flows    or torch.CudaTensor()
   local sz = sample.input:size()
   local n, c, h, w = table.unpack(sample.input:size():totable())

   self.ref_imgs:resize(n, 3, h, w):copy(sample.input[{{},{1,3},{},{}}])
   self.tar_imgs:resize(n, 3, h, w):copy(sample.input[{{},{4,6},{},{}}])
   self.masks:resize(n, h, w):copy(sample.masks)
   self.rhos:resize( n, h, w):copy(sample.rhos)
   self.flows:resize(n, 3, h, w):copy(sample.flows)
   if self.opt.in_trimap then
       self.trimaps = self.trimaps or torch.CudaTensor()
       self.trimaps:resize(n, 1, h, w):copy(sample.trimaps)
   end
end

function Trainer:copyInputsMultiScale(sample)
   local multiscale_in = {self.ref_imgs, self.tar_imgs, self.rhos, self.masks, self.flows}

   local multiscale_out = self.multiScaleData:forward(multiscale_in)
   self.multi_ref_imgs  = multiscale_out[1]
   self.multi_tar_imgs  = multiscale_out[2]
   self.multi_rhos      = multiscale_out[3]
   self.multi_masks     = multiscale_out[4]
   self.multi_flows     = multiscale_out[5]

   for i = 1, #self.multi_flows do
       -- Rescale the loss weight for flow in different scale
       local ratio = 2^(#self.multi_flows - i)
       self.multi_flows[i]:narrow(2,3,1):mul(ratio)
   end
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local ratio = (epoch >= self.opt.lr_decay_start and epoch % self.opt.lr_decay_step == 0) and 0.5 or 1.0
   return self.optimState.learningRate * ratio
end

return M.Trainer
