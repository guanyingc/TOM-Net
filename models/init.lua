require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
   -- LOAD OR CREATE THE MODEL 
   local model
   if checkpoint then -- Resume
       model   = checkpoint.model
   elseif opt.retrain ~= 'none' then
       assert(paths.filep(opt.retrain), 'Model not found: ' .. opt.retrain)
       print('=> [Retrain] Loading model from ' .. opt.retrain)
       model   = torch.load(opt.retrain).model:cuda()
   else
       print('=> Creating model from: models/' .. opt.networkType .. '.lua')
       model = require('models/' .. opt.networkType)(opt)
   end
   -- SET THE CUDNN FLAGS
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end    

   model:cuda()
   cudnn.convert(model, cudnn)
   return model
end

return M
