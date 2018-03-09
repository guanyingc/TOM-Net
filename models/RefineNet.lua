require 'nngraph'
local model_utils = require 'models/model_utils'

local Conv = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local w, h

function normalizeInput()
    local normalize = nn.ParallelTable()
    normalize:add(nn.Identity())  -- Image
    normalize:add(nn.MulConstant(1/384, false)) -- flow
    normalize:add(nn.SoftMax())   -- Mask
    normalize:add(nn.Identity())  -- Rho
    return normalize
end

local function downsampleBlock(cin, cout, k, down_num)
    local downsample = nn.Sequential()
    local pad = (k - 1) / 2
    downsample:add(nn.SpatialReflectionPadding(pad, pad, pad, pad))
    downsample:add(Conv(cin, cout, k, k, 1, 1, 0, 0))
    downsample:add(BatchNorm(cout))
    downsample:add(nn.ReLU(true))

    for i = 1, down_num do
        print('Adding downsample block')
        downsample:add(Conv(cout, cout, 4, 4, 2, 2, 1, 1))
        downsample:add(BatchNorm(cout))
        downsample:add(nn.ReLU(true))
    end
    return downsample
end

local function upsampeBlock(cin,  down_num)
    local upsample = nn.Sequential()
    local k = 3
    local pad = (k - 1) / 2
    for i = 1, down_num do
        upsample:add(nn.SpatialFullConvolution(cin, cin, 4, 4, 2, 2, pad, pad))
        upsample:add(BatchNorm(cin))
        upsample:add(nn.ReLU(true))
    end
    upsample:add(Conv(cin, cin, 3, 3, 1, 1, pad, pad))
    upsample:add(BatchNorm(cin))
    upsample:add(nn.ReLU(true))
    return upsample
end

local function resNetBlock(filter_sz, state_num, in_channel)
    local function shortcut(str)
        if str == 1 then 
            return nn.Identity()
        else
            return nn.Sequential():add(Average(1,1))
        end
    end
    local pad = (filter_sz - 1) / 2
    local str = 1
    local block = nn.Sequential()
    local path = nn.Sequential()
        :add(Conv(in_channel, state_num, filter_sz, filter_sz, str, str, pad, pad))
        :add(BatchNorm(state_num))
        :add(nn.ReLU(true))
        :add(Conv(state_num,  state_num, filter_sz, filter_sz, str, str, pad, pad))
        :add(BatchNorm(state_num))
    local concat = nn.ConcatTable()
        :add(path)
        :add(shortcut(1))
    block:add(concat):add(nn.CAddTable(true))
    return block
end

function createResNet(layers_num, filter_sz, in_channel, state_num)
    local model = nn.Sequential()
    local pad   = (filter_sz - 1) / 2
    local str   = 1

    for layer = 1, (layers_num - 2) / 2 do
        print('Adding residual block')
        model:add(resNetBlock(filter_sz, state_num, state_num))
    end
    return model
end

local function createModel(opt)
    w = opt.crop_w
    h = opt.crop_h
 
    local layers    = opt.layers_num
    local state_num = opt.channel_sz
    local k         = opt.filter_sz
    local c_in      = 8
    local in_img    = -nn.Identity() 
    local in_flow   = -nn.Identity();
    local in_mask   = -nn.Identity();
    local in_rho    = -nn.Identity();
    local input     = {in_img, in_flow, in_mask, in_rho}

    local net_input = input     - normalizeInput() - nn.JoinTable(2)
    local down_in   = net_input - downsampleBlock(c_in, state_num, 9, opt.down_num)
    local feature   = down_in   - createResNet(layers, k, state_num, state_num)

    local flow_feat = feature   - upsampeBlock(state_num, opt.down_num)
    local flow      = {flow_feat, in_flow} - nn.JoinTable(2) - Conv(state_num+2, 2, 3, 3, 1, 1, 1, 1)

    local rho_feat  = feature - upsampeBlock(state_num, opt.down_num)
    local rho       = {rho_feat, in_rho} - nn.JoinTable(2) - Conv(state_num+1, 1, 3, 3, 1, 1, 1, 1)
    local output    = {flow, rho}
    local net       = nn.gModule(input, output)
    return net
end
return createModel
