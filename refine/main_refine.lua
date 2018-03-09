-----------------------------------------------------------------------------
-- IMPORT DIFFERENT MODULE, OPTIONS, DATALOADER, CHECKPOINTS, MODEL, TRAINER
-----------------------------------------------------------------------------
require 'torch'
require 'paths'

local opts_ref      = require 'refine/opts_refine'; local opt_ref = opts_ref.parse(arg)
local DataLoader    = require 'dataloader'
local checkpoints   = require 'checkpoints'
local models        = require 'models/init'
local Trainer       = require 'train'

local dict_utils    = require 'utils/dict_utils'
local io_utils      = require 'utils/io_utils'
local eval_utils    = require 'eval/eval_utils'

-----------------------------------------------------------------------------
-- INTIALIZATION
-----------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(opt_ref.manualSeed)
cutorch.manualSeedAll(opt_ref.manualSeed)

local trainLoader, valLoader = DataLoader.create(opt_ref)
local checkp, optimState     = checkpoints.latest(opt_ref)
local model                  = models.setup(opt_ref, checkp)
local trainer                = Trainer(model, opt_ref, optimState)

local predictor = eval_utils.getModel(opt_ref.coarse_net, true, true)

if opt_ref.valOnly then
    local results= trainer:test(1, valLoader, 'val', predictor)
    return
end

-----------------------------------------------------------------------------
-- CONFIGURE START POINTS AND HISTORY
-----------------------------------------------------------------------------
local train_hist = io_utils.loadt7(checkp, paths.concat(opt_ref.resume, 'train_hist.t7'))
local val_hist   = io_utils.loadt7(checkp, paths.concat(opt_ref.resume, 'val_hist.t7'))  
local startEpoch = checkp and checkp.epoch + 1 or opt_ref.startEpoch

local function add_history(epoch, history, split)
    if split == 'train' then
        train_hist = dict_utils.insertSubDicts(train_hist, history)
        torch.save(paths.concat(opt_ref.save, split .. '_hist.t7'), train_hist)
    elseif split == 'val' then
        val_hist = dict_utils.insertSubDicts(val_hist, history)
        torch.save(paths.concat(opt_ref.save, split .. '_hist.t7'), val_hist)
    else
        error(string.format('Unknown split: %s', split))
    end
end

for epoch = startEpoch, opt_ref.nEpochs do
   -- TRAIN FOR A SINGLE EPOCH
   local trainLoss = trainer:train(epoch, trainLoader, 'train', predictor)

   -- SAVE CHECKPOINTS
   if (epoch % opt_ref.saveInterval == 0) then
       print(string.format("\t**** Epoch %d saving checkpoint ****", epoch))
       checkpoints.save(opt_ref, model, trainer.optimState, epoch)
   end
   -- SAVE AND PLOT RESULTS FOR TRAINING STAGE
   add_history(epoch, trainLoss, 'train')
   io_utils.plot_results_compact(train_hist, opt_ref.logDir, 'train')

   -- VALIDATION ON SYNTHETIC DATA
   if (epoch % opt_ref.val_interval == 0) then
      local valResult = trainer:test(epoch, valLoader, 'val', predictor)
      add_history(epoch, valResult, 'val')
      io_utils.plot_results_compact(val_hist, opt_ref.logDir, 'val')
   end
   
   collectgarbage()
end

