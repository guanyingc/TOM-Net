local M = {}
local str_utils = require 'utils/str_utils'
function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text('Torch-7 TOM-Net')
    ----------- Dataset Options -----------
    cmd:option('-dataset',       'TOMDataset',            'Data: Transparent object matting')
    cmd:option('-data_dir',      'data/datasets/TOM-Net_Synth_Train_178k')
    cmd:option('-train_list',    'train_simple_98k.txt',  'Data: Train list')
    cmd:option('-val_list',      'val_imglist.txt',       'Data: Val list')
    cmd:option('-data_aug',       true,                   'Data: data augmentation')
    cmd:option('-scale_h',        512,                    'Data: rescale height')
    cmd:option('-scale_w',        512,                    'Data: rescale width')
    cmd:option('-crop_h',         448,                    'Data: crop height')
    cmd:option('-crop_w',         448,                    'Data: crop width')
    cmd:option('-noise',          0.05,                   'Data: noisy level')
    cmd:option('-rot_ang',         0.3,                   'Data: rotate data')
    cmd:option('-max_image_num',  -1,                     'Data: >0 for max numbers')
     ------------ Device Options --------------------
    cmd:option('-manualSeed',      0,          'Devices: manually set RNG seed')
    cmd:option('-cudnn',          'fastest',   'Devices: fastest|default|deterministic')
    cmd:option('-nThreads',        8,          'Devices: number of data loading threads')
    ------------- Training Options ----------  ----------
    cmd:option('-startEpoch',      1,          'Epoch: Manual start epoch for restart')
    cmd:option('-nEpochs',         20,         'Epoch: Number of total epochs to run')
    cmd:option('-batchSize',       4,          'Epoch: mini-batch size')
    cmd:option('-lr',              1e-4,       'LR: initial learning rate')
    cmd:option('-lr_decay_start',  10,         'LR: number of epoch when lr start to decay')
    cmd:option('-lr_decay_step',   5,          'LR: step for the lr decay')
    cmd:option('-solver',         'ADAM',      'Solver: ADAM only')
    cmd:option('-beta_1',          0.9,        'Solver: first param of Adam optimizer')
    cmd:option('-beta_2',          0.999,      'Solver: second param of Adam optimizer')
    ------------ Network Options ----------------------
    cmd:option('-networkType',    'CoarseNet', 'Network: version')
    cmd:option('-use_BN',          true,       'Network: Batch norm')
    cmd:option('-ms_num',           4,         'Multiscale: scales level')
    ----------- Checkpoint options ---------------
    cmd:option('-resume',         'none',      'Checkpoint: Reload checkpoint and state')
    cmd:option('-retrain',        'none',      'Checkpoint: Reload checkpoint only')
    cmd:option('-suffix',          '',         'Checkpoint: checkpoint suffix')
    cmd:option('-saveInterval',    1,          'Checkpoint: epochs to save checkpoints (overwrite)')
    cmd:option('-save_new',        1,          'Checkpoint: epochs to save new checkpoints')
    -------------- Loss Options -----------
    cmd:option('-flow_w',          0.01,       'Loss: Flow weight')
    cmd:option('-img_w',           1,          'Loss: Image reconstruction weight')
    cmd:option('-mask_w',          0.1,        'Loss: Mask weight')
    cmd:option('-rho_w',           1,          'Loss: Attenuation mask weight')
    ------------- Display Options -------------
    cmd:option('-train_display',   20,         'Display: Iteration to display train loss')
    cmd:option('-train_save',      300,        'Display: Iteration to save train results ')
    cmd:option('-val_interval',    1,          'Display: Intervals to do the validation')
    cmd:option('-val_display',     5,          'Display: Iteration to display val loss')
    cmd:option('-val_save',        5,          'Display: Iteration to save val results ')
    cmd:option('-valOnly',         false,      'Display: Run on validation set only')
    --------------- Log Options --------
    cmd:option('-prefix',          '',         'Log: prefix of the log directory' )
    cmd:option('-debug',          false,       'Log: debug mode' )
    local opt = cmd:parse(arg or {})

    opt.startTime = os.time()
    opt.logfile = M.checkpath(opt)
    cmd:log(opt.logfile, opt)
    return opt
end

function M.getSaveDirName(opt)
    opt.date, opt.time = str_utils.getDateTime()
    local dName = string.format('%s_%s_', opt.date, opt.prefix) .. opt.networkType
    local params = {'scale_h', 'crop_h'}
    for k, v in pairs(params) do
        dName = dName .. string.format('_%s-%d', v, opt[v]) 
    end
    local params = {'flow_w', 'mask_w', 'rho_w', 'img_w'}
    for k, v in pairs(params) do
        dName = dName .. string.format('_%s-%.3f', v, opt[v]) 
    end
    local params = {'lr'}
    for k, v in pairs(params) do
        dName = dName .. string.format('_%s-%f', v, opt.lr)
    end
    dName = dName .. (opt.retrain ~= 'none' and '_retrain' or '')
    dName = dName .. (opt.resume  ~= 'none' and '_resume' or '')
    dName = dName .. (opt.valOnly   and '_valOnly' or '')

    if opt.debug then
        dName = string.format('%s_%s_debug', opt.date, opt.prefix)
        opt.max_image_num = 10
        opt.train_save = 1
        opt.train_display = 1
        opt.val_save = 100
    end
    local logDir = paths.concat('data/training', dName, 'logdir')
    local save   = paths.concat('data/training', dName, 'checkpointdir')
    return logDir, save, dName
end

function M.checkpath(opt)
    opt.logDir, opt.save, opt.dirName = M.getSaveDirName(opt)
    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then 
        error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end
    if not paths.dirp(opt.logDir) and not paths.mkdir(opt.logDir) then
    	error('error: unable to create logdir: ' .. opt.logDir .. '\n')
    end
    for i,d in pairs({'/train/Images', '/val/Images'}) do
        if not paths.dirp(opt.logDir .. d) then
            paths.mkdir(opt.logDir .. d)
        end
    end
    logfile = paths.concat(opt.logDir, string.format('%s_%s_%s', opt.date, opt.time, opt.dirName))
    return logfile
end

return M
