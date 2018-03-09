require 'torch'
require 'nn'
require 'criterion/EPECriterion'

TOMCriterionFlow, parent = torch.class('nn.TOMCriterionFlow', 'nn.Criterion')

function TOMCriterionFlow:__init(opt) 
    self.flow_w    = opt.flow_w
    self.flow_loss = opt.flow_loss
    self.refine    = opt.refine

    local flow_crit = nn.EPECriterion()

    if self.refine then
        print(string.format('[Flow Sup Criterion]: single scale'))
        self.flow_crits = flow_crit
    else
        self.ms_num     = opt.ms_num
        self.flow_crits = nn.ParallelCriterion()
        for i = 1, self.ms_num do 
            print(string.format('[Flow Sup Criterion]: scale %d/%d', i, self.ms_num))
            self.flow_crits:add(flow_crit:clone(), self.flow_w)
        end
    end
    self.flow_crit_pred   = {}
    self.flow_crit_target = {}
    self.smo_crit_target  = {}
end

function TOMCriterionFlow:updateOutput(pred, target)
    if self.refine then
        local mask    = target:narrow(2, 3, 1)
        local gt_flow = target:narrow(2, 1, 2)
        mask = mask:expandAs(gt_flow)
        self.flow_crit_pred   = torch.cmul(pred, mask)
        self.flow_crit_target = torch.cmul(gt_flow, mask)
    else
        for i = 1, self.ms_num do 
            local mask    = target[i]:narrow(2, 3, 1)
            local gt_flow = target[i]:narrow(2,1,2)
            mask = mask:expandAs(gt_flow)
            self.flow_crit_pred[i]   = torch.cmul(pred[i], mask)
            self.flow_crit_target[i] = torch.cmul(gt_flow, mask)
        end
    end
    local losses = {}
    self.flow_crits:forward(self.flow_crit_pred, self.flow_crit_target)
    if self.refine then
        losses['loss_flow'] = self.flow_crits.output * self.flow_w
        self.epe = losses['loss_flow'] / self.flow_w
    else
        for i = 1, self.ms_num do
            losses['sc' .. i .. '_flow'] = self.flow_crits.criterions[i].output * self.flow_w
            if i == self.ms_num then
                self.epe = losses['sc' .. i .. '_flow'] / self.flow_w
            end
        end
    end
    return losses
end

function TOMCriterionFlow:updateGradInput(pred, target)
    self.grad = self.flow_crits:backward(self.flow_crit_pred, self.flow_crit_target)
    return self.grad 
end

function TOMCriterionFlow:cuda()
    self.flow_crits:cuda() 
end
