require 'torch'
require 'nn'

TOMCriterionUnsup, parent = torch.class('nn.TOMCriterionUnsup', 'nn.Criterion')

function TOMCriterionUnsup:__init(opt) 
    self.refine    = opt.refine
    self.rho_w     = opt.rho_w
    local rho_crit = nn.MSECriterion()

    self.crit_container = nn.ParallelCriterion()
    if self.refine then
        print(string.format('\t Refine stage: adding rho loss'))
        self.crit_container:add(rho_crit, self.rho_w)
    else
        local rec_crit  = nn.MSECriterion()
        local mask_crit = cudnn.SpatialCrossEntropyCriterion()
        self.ms_num     = opt.ms_num
        self.img_w      = opt.img_w
        self.mask_w     = opt.mask_w
        for i = 1, self.ms_num do
            print(string.format('[Unsup Criterion]: scale %d/%d', i, self.ms_num))
            local multi_crits = nn.ParallelCriterion()
            multi_crits:add(rec_crit:clone(),  self.img_w); 
            multi_crits:add(mask_crit:clone(), self.mask_w);
            multi_crits:add(rho_crit:clone(),  self.rho_w); 
            self.crit_container:add(multi_crits)
        end
    end
end

function TOMCriterionUnsup:updateOutput(pred, target)
    local loss   = self.crit_container:forward(pred, target)
    local losses = {}
    local crits  = self.crit_container.criterions
    if self.refine then
        losses['loss_rho'] = crits[1].output * self.rho_w
    else
        for i = 1, self.ms_num do
          losses['sc' .. i .. '_img']  = crits[i].criterions[1].output * self.img_w
          losses['sc' .. i .. '_mask'] = crits[i].criterions[2].output * self.mask_w
          losses['sc' .. i .. '_rho']  = crits[i].criterions[3].output * self.rho_w
        end
    end
    return losses
end

function TOMCriterionUnsup:updateGradInput(pred, target)
  self.grad = self.crit_container:backward(pred, target)
  return self.grad
end

function TOMCriterionUnsup:cuda()
  self.crit_container:cuda() 
end

