require 'nngraph'
local model_utils = require 'models/model_utils'

local Conv = nn.SpatialConvolution
local Upsample = nn.SpatialUpSamplingNearest--nn.SpatialUpSamplingBilinear
local w, h

local function createOutput(cin, scale)
    local output = nn.Sequential()
    output:add(nn.JoinTable(2))
    local multi_task = nn.ConcatTable()
    local k   = 3
    local str = 1
    local pad = (k - 1) / 2

    local flow_path = nn.Sequential():add(Conv(cin, 2, k, k, str, str, pad, pad))
    local ratio = w / 2^(scale-1)
    flow_path:add(nn.Tanh()):add(nn.MulConstant(ratio, false))
    multi_task:add(flow_path) -- Flow

    multi_task:add(Conv(cin, 2, k, k, str, str, pad, pad)) -- Mask

    local rho_branch = nn.Sequential():add(Conv(cin, 1, k, k, str, str, pad, pad))
    multi_task:add(rho_branch) -- Rho
    output:add(multi_task)
    return output
end

function normalizeOutput(scale)
    local output    = nn.Sequential()
    local normalize = nn.ParallelTable()

    local flow_branch = nn.Sequential()
    local ratio = 2.0^(scale-1) / w
    flow_branch:add(nn.MulConstant(ratio, false))

    normalize:add(flow_branch)   -- Flow
    normalize:add(nn.SoftMax())  -- Mask
    normalize:add(nn.Identity()) -- Rho

    output:add(normalize)
    output:add(nn.JoinTable(2))
    output:add(Upsample(2))
    return output
end

local function createModel(opt)
    w = opt.crop_w
    h = opt.crop_h
    local use_BN   = opt.use_BN
    local encoderB = model_utils.encoderB
    local decoderB = model_utils.decoderB

    local c_in = 3
    if opt.in_trimap then
        c_in = c_in + 1
    elseif opt.in_bg then
        c_in = c_in + 3
    end
    local c_0 = 16;  local c_1 = 16;  local c_2 = 32;  local c_3 = 64;   
    local c_4 = 128; local c_5 = 256; local c_6 = 256; local c_7 = 256;

    local input = -nn.Identity() -- 384*512
    ---- Encoder
    local conv0 = input
            - encoderB(c_in,c_0,3,1,use_BN)
            - encoderB(c_0,c_0,3,1, use_BN)
    local conv1 = conv0 
            - encoderB(c_0,c_1,3,2, use_BN)
            - encoderB(c_1,c_1,3,1, use_BN)
    local conv2 = conv1 
            - encoderB(c_1,c_2,3,2, use_BN)
            - encoderB(c_2,c_2,3,1, use_BN)
    local conv3 = conv2 
            - encoderB(c_2,c_3,3,2, use_BN)
            - encoderB(c_3,c_3,3,1, use_BN)
    local conv4 = conv3
            - encoderB(c_3,c_4,3,2, use_BN)
            - encoderB(c_4,c_4,3,1, use_BN)
    local conv5 = conv4
            - encoderB(c_4,c_5,3,2, use_BN)
            - encoderB(c_5,c_5,3,1, use_BN)
    local conv6 = conv5
            - encoderB(c_5,c_6,3,2, use_BN)
            - encoderB(c_6,c_6,3,1, use_BN)
    
    --- DECODER 
    local deconv6 = {}
    local deconv5 = {}
    local deconv4 = {}
    local deconv3 = {}
    local deconv2 = {}
    local deconv1 = {}
    local outputs = {}
    local n_out = 3     -- Num of output branches (flow, mask, rho)
    local c_out_num = 5 -- Total num of channels  (2 + 2 + 1)

    for i = 1, n_out do
        deconv6[i] = conv6 - decoderB(c_6,c_5,3,1, true, use_BN)
    end
    deconv6[n_out+1] = conv5

    for i = 1, n_out do -- Deconv 5: 24*32
        deconv5[i] = deconv6 - decoderB((n_out+1)*c_5,c_4,3,1, false, use_BN)
    end
    deconv5[n_out+1] = conv4
      
    for i = 1, n_out do -- Deconv 4: 48*64
        deconv4[i] = deconv5 - decoderB((n_out+1)*c_4,c_3,3,1, false, use_BN)
    end
    deconv4[n_out+1] = conv3

    local idx = 1
    local c_out = 0; 

    if opt.ms_num >= 5 then -- Scale 5 output 24 * 32
        local s5_out     = deconv5 - createOutput((n_out+1)*c_4+c_out, 5)
        local s5_out_up  = s5_out  - normalizeOutput(5)
        deconv4[n_out+2] = s5_out_up
        outputs[idx]     = s5_out
        idx   = idx + 1
        c_out = c_out_num
    end

    for i = 1, n_out do  -- Deconv 3: 96*128
        deconv3[i] = deconv4 - decoderB((n_out+1)*c_3+c_out,c_2,3,1, false, use_BN)
    end
    deconv3[n_out+1] = conv2

    if opt.ms_num >= 4 then -- Scale 4 output 48*64
        local s4_out     = deconv4 - createOutput((n_out+1)*c_3+c_out, 4)
        local s4_out_up  = s4_out  - normalizeOutput(4)
        deconv3[n_out+2] = s4_out_up
        outputs[idx]     = s4_out
        idx   = idx + 1
        c_out = c_out_num
    end

    for i = 1, n_out do -- Deconv 2: 192*256
        deconv2[i] = deconv3 - decoderB((n_out+1)*c_2+c_out,c_1,3,1, false, use_BN)
    end
    deconv2[n_out+1] = conv1

    if opt.ms_num >= 3 then -- Scale 3 output 96 * 128
        local s3_out     = deconv3 - createOutput((n_out+1)*c_2+c_out, 3)
        local s3_out_up  = s3_out  - normalizeOutput(3)
        deconv2[n_out+2] = s3_out_up
        outputs[idx]     = s3_out
        idx   = idx + 1
        c_out = c_out_num
    end

    for i = 1, n_out do -- Deconv 1: 384*512
        deconv1[i] = deconv2 - decoderB((n_out+1)*c_1+c_out,c_0,3,1, false, use_BN)
    end
    deconv1[n_out+1] = conv0

    if opt.ms_num >= 2 then -- Scale 2 output
        local s2_out     = deconv2 - createOutput((n_out+1)*c_1+c_out, 2)
        local s2_out_up  = s2_out  - normalizeOutput(2)
        deconv1[n_out+2] = s2_out_up
        outputs[idx]     = s2_out
        idx   = idx + 1
        c_out = c_out_num
    end

    local s1_out = deconv1 - createOutput((n_out+1)*c_0+c_out, 1)
    outputs[idx] = s1_out

    local net = nn.gModule({input},outputs)
    return net
end
return createModel
