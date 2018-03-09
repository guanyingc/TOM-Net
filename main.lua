-----------------------------------------------------------------------------
-- IMPORT DIFFERENT MODULE, OPTIONS, DATALOADER, CHECKPOINTS, MODEL, TRAINER
-----------------------------------------------------------------------------
require 'torch'
require 'paths'

local opts          = require 'opts'; local opt = opts.parse(arg)
local DataLoader    = require 'dataloader'
local checkpoints   = require 'checkpoints'
local models        = require 'models/init'
local Trainer       = require 'train'

local dict_utils    = require 'utils/dict_utils'
local io_utils      = require 'utils/io_utils'

-----------------------------------------------------------------------------
-- INTIALIZATION
-----------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

local trainLoader, valLoader = DataLoader.create(opt)
local checkp, optimState     = checkpoints.latest(opt)
local model                  = models.setup(opt, checkp)
local trainer                = Trainer(model, opt, optimState)

if opt.valOnly then
    local results= trainer:test(1, valLoader, 'val')
    return
end
-----------------------------------------------------------------------------
-- CONFIGURE START POINTS AND HISTORY
-----------------------------------------------------------------------------
local train_hist = io_utils.loadt7(checkp, paths.concat(opt.resume, 'train_hist.t7'))
local val_hist   = io_utils.loadt7(checkp, paths.concat(opt.resume, 'val_hist.t7'))  
local startEpoch = checkp and checkp.epoch + 1 or opt.startEpoch

local function add_history(epoch, history, split)
    if split == 'train' then
        train_hist = dict_utils.insertSubDicts(train_hist, history)
        torch.save(paths.concat(opt.save, split .. '_hist.t7'), train_hist)
    elseif split == 'val' then
        val_hist = dict_utils.insertSubDicts(val_hist, history)
        torch.save(paths.concat(opt.save, split .. '_hist.t7'), val_hist)
    else
        error(string.format('Unknown split: %s', split))
    end
end

-----------------------------------------------------------------------------
---- START TRAINING
-----------------------------------------------------------------------------
for epoch = startEpoch, opt.nEpochs do

   -- TRAIN FOR A SINGLE EPOCH
   local trainLoss = trainer:train(epoch, trainLoader, 'train')

   -- SAVE CHECKPOINTS
   if (epoch % opt.saveInterval == 0) then
       print(string.format("\t**** Epoch %d saving checkpoint ****", epoch))
       checkpoints.save(opt, model, trainer.optimState, epoch)
   end
   -- SAVE AND PLOT RESULTS FOR TRAINING STAGE
   add_history(epoch, trainLoss, 'train')
   io_utils.plot_results_compact(train_hist, opt.logDir, 'train')

   -- VALIDATION ON SYNTHETIC DATA
   if (epoch % opt.val_interval == 0) then
      local valResult = trainer:test(epoch, valLoader, 'val')
      add_history(epoch, valResult, 'val')
      io_utils.plot_results_compact(val_hist, opt.logDir, 'val')
   end
   
   collectgarbage()
end

