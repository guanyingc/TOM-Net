--[[
     File Name           :     checkpoints.lua
     Created By          :     Chen Guanying (GoYchen@foxmail.com)
     Creation Date       :     [2018-03-03 20:05]
     Last Modified       :     [2018-03-05 16:22]
     Description         :      
--]]
--

--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end
   local suffix = opt.suffix -- If specify checkpoint epoch number
   if opt.suffix == '' then
       local f = io.open(paths.concat(opt.resume, 'latest'), 'r') 
       suffix = f:read()
   end
   local checkpointPath = paths.concat(opt.resume, 'checkpoint' .. suffix .. '.t7')
   local optimStatePath = paths.concat(opt.resume, 'optimState' .. suffix .. '.t7')
   assert(paths.filep(checkpointPath), 'Saved model not found: ' .. checkpointPath)
   assert(paths.filep(optimStatePath), 'Saved optimState not found: ' .. optimStatePath)

   print('=> [Resume] Loading Checkpoint' .. checkpointPath)
   print('=> [Resume] Loading Optim state' .. optimStatePath)
   local checkpoint = torch.load(checkpointPath)
   local optimState = torch.load(optimStatePath)
   return checkpoint, optimState
end

function checkpoint.save(opt, model, optimState, epoch)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end
   -- create a clean copy on the CPU without modifying the original network
   model = model:clearState()

   local checkpoint = {}
   checkpoint.opt = opt
   checkpoint.epoch = epoch
   checkpoint.model = model
   local suffix 
   if opt.save_new > 0 then
       epoch_num = math.floor((epoch-1)/opt.save_new) * opt.save_new + 1
       suffix = tostring(epoch_num)
   else
       suffix = ''
   end
   torch.save(paths.concat(opt.save, 'checkpoint' .. suffix .. '.t7'), checkpoint)
   torch.save(paths.concat(opt.save, 'optimState' .. suffix .. '.t7'), optimState)

   fout = io.open(paths.concat(opt.save, 'latest'), 'w') 
   fout:write(string.format('%d\n', suffix))
   fout:close()
end

return checkpoint
