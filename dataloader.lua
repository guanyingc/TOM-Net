--[[
     File Name           :     dataloader.lua
     Created By          :     Chen Guanying (GoYchen@foxmail.com)
     Creation Date       :     [2018-03-03 20:02]
     Last Modified       :     [2018-03-03 20:02]
     Description         :      
--]]

--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('TOM.DataLoader', M)

function DataLoader.create(opt)
   local loaders = {}
   local datasets
   datasets = require 'datasets/TOMDataset'

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end
   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize)
   self.in_trimap = opt.in_trimap
   self.split = split
end

function DataLoader:batch_size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:size()
   return self.__size
end

function DataLoader:run(split, maxNumber)
   local split = split or 'train'

   local threads = self.threads
   local in_trimap = self.in_trimap
   local fullSize, batchSize = self.__size, self.batchSize
   local size = (maxNumber ~= nil and maxNumber > 0) and math.min(maxNumber, fullSize) or fullSize
   local perm = (split == 'val') and torch.range(1, size) or torch.randperm(size)
   print(string.format('[Dataloader run] split: %s, size: %d/%d ', split, size, fullSize))

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local batch, masks, rhos, flows, trimaps, imageSize
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = sample.input
                  if not batch then
                     imageSize = input:size():totable()
                     batch = torch.FloatTensor(sz, table.unpack(imageSize))
                     masks = torch.FloatTensor(sz, input:size()[2], input:size()[3])
                     rhos  = torch.FloatTensor(sz, input:size()[2], input:size()[3])
                     flows = torch.FloatTensor(sz, 3, input:size()[2], input:size()[3])
                     if in_trimap then
                        trimaps = torch.FloatTensor(sz, input:size()[2], input:size()[3])
                     end
                  end
                  batch[i]:copy(input)
                  masks[i]:copy(sample.mask)
                  rhos[i]:copy(sample.rho)
                  flows[i]:copy(sample.flow)
                  if in_trimap then
                      trimaps[i]:copy(sample.trimap)
                  end
               end
               collectgarbage()
               local batch_sample = {
                  input = batch,
                  masks = masks,
                  rhos  = rhos,
                  flows = flows
               }
               if in_trimap then
                  batch_sample.trimaps = trimaps
               end
               return batch_sample
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
