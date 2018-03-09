require 'nn'
require 'stn'
require 'models/AffineGridGeneratorUSOF'
local model_utils = {}

local Conv = nn.SpatialConvolution
local Upsample =  nn.SpatialUpSamplingNearest -- nn.SpatialUpSamplingBilinear
local BatchNorm = nn.SpatialBatchNormalization
local Activate = nn.ReLU

function model_utils.encoderB(cin, cout, k, str, use_BN)
    local pad = math.floor((k-1)/2)
    local encoderB = nn.Sequential()
    encoderB:add(Conv(cin,cout,k,k,str,str,pad,pad))
    if use_BN == true then 
        print('[Netowrk] Use batchnorm in encoder')
        encoderB:add(BatchNorm(cout))
    end
    encoderB:add(Activate(true))
    return encoderB
end

function model_utils.decoderB(cin, cout, k, str, bottom, use_BN)
    local bottom = bottom or false
    local pad = math.floor((k-1)/2)
    local decoderB = nn.Sequential()
    if not bottom then
        decoderB:add(nn.JoinTable(2))
    end
    decoderB:add(Conv(cin,cout,k,k,str,str,pad,pad))
    if use_BN then
        print('[Netowrk] Use batchnorm in decoder block')
        decoderB:add(BatchNorm(cout))
    end
    decoderB:add(Activate(true))
    decoderB:add(Upsample(2))
    return decoderB
end

function model_utils.createFlowPooling(scale)
    local flow_scale = nn.Sequential()
    flow_scale:add(nn.SpatialAveragePooling(scale, scale, scale, scale))
    flow_scale:add(nn.MulConstant(1/scale, true))
    return flow_scale
end

function model_utils.createMultiScaleData(opt)
    local multiScale  = nn.ParallelTable()
    local avgPooling  = nn.ConcatTable()
    local maxPooling  = nn.ConcatTable()
    local flowPooling = nn.ConcatTable()
    for i = opt.ms_num, 1, -1 do
        local scale = 2^(i-1)
        maxPooling:add(nn.SpatialMaxPooling(scale, scale, scale, scale))
        avgPooling:add(nn.SpatialAveragePooling(scale, scale, scale, scale))
        flowPooling:add(model_utils.createFlowPooling(scale)) 
    end
    multiScale:add(avgPooling:clone()) -- ref
    multiScale:add(avgPooling:clone()) -- tar
    multiScale:add(avgPooling:clone()) -- rho
    multiScale:add(maxPooling:clone()) -- mask
    multiScale:add(flowPooling)
    return multiScale
end

function model_utils.createMultiScaleWarping(ms_num)
    local warping_module = nn.ConcatTable()
    for i = 1, ms_num do
        local single_warping = nn.Sequential()
        local select_scale   = nn.ParallelTable()
        select_scale:add(nn.SelectTable(i))
        select_scale:add(nn.SelectTable(i))
        single_warping:add(select_scale)
        single_warping:add(model_utils.createSingleWarpingModule())
        warping_module:add(single_warping)
    end
    return warping_module
end

function model_utils.createSingleWarpingModule()
    local warping_module = nn.Sequential()
    local parallel_1     = nn.ParallelTable()
    local trans          = nn.Sequential()
    trans:add(nn.Identity()):add(nn.Transpose({2,3},{3,4}))
    parallel_1:add(trans)
    parallel_1:add(nn.AffineGridGeneratorUSOF())
    warping_module:add(parallel_1)
    warping_module:add(nn.BilinearSamplerBHWD())
    warping_module:add(nn.Transpose({3,4},{2,3}))
    return warping_module
end
return model_utils
