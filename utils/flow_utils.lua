require 'torch'
require 'math'
require 'paths'
local flow_utils = {}
local timer = torch.Timer()

-- Simple helper functions for refractive flow field
function flow_utils.cal_epe_flow(gt_flow, pred_flow, valid, mask)
    -- For single image
    local buffer = gt_flow:clone()
    if pred_flow == false then
        buffer:pow(2)
    else
        buffer:add(-1*pred_flow):pow(2)
    end
    buffer = buffer:sum(1):sqrt()
    buffer:cmul(valid:typeAs(buffer))
    local error_all = buffer:sum() / valid:sum()
    local error_roi = torch.cmul(buffer, mask):sum() / mask:sum()
    return error_all, error_roi, buffer
end

function flow_utils.writeShortFlowFile(filename, F)
    F = F:short():permute(2,3,1):clone()
    TAG_FLOAT = 202021.25 
    local ff = torch.DiskFile(filename, 'w'):binary()
    ff:writeFloat(TAG_FLOAT)
    ff:writeInt(F:size(2)) -- width
    ff:writeInt(F:size(1)) -- height
    ff:writeShort(F:storage())
    ff:close()
end

function flow_utils.loadShortFlowFile(filename)
    TAG_FLOAT = 202021.25 
    local ff = torch.DiskFile(filename):binary()
    local tag = ff:readFloat()
    if tag ~= TAG_FLOAT then
      xerror('unable to read '..filename..  ' perhaps bigendian error','readflo()')
    end
    local w = ff:readInt()
    local h = ff:readInt()
    local nbands = 2
    local tf = torch.ShortTensor(h, w, nbands)
    ff:readShort(tf:storage())
    ff:close()

    local flow = tf:permute(3,1,2):float()
    return flow
end

function flow_utils.flowToColor(flow)
    local flow = flow:float()
    local F_val
    if flow:size(1) == 3 then
        F_val = flow[{{3}, {}, {}}]:ge(0.1):float()
    else
        F_val = torch.Tensor(flow:size(2), flow:size(3)):fill(1)
    end
    local F_du  = flow[{ {2},{},{} }]:clone()
    local F_dv  = flow[{ {1},{},{} }]:clone()
    local u_max = F_du:maskedSelect(F_val:byte()):abs():max()
    local v_max = F_dv:maskedSelect(F_val:byte()):abs():max()
    local F_max = math.max(u_max, v_max)
    local F_mag = torch.sqrt(torch.pow(F_du,2):add(torch.pow(F_dv,2)))                                            
    local F_dir = torch.atan2(F_dv, F_du)
    local img   = flow_utils.flow_map(F_mag, F_dir, F_val, F_max, 8)
    return img
end

function flow_utils.flow_map(F_mag, F_dir, F_val, F_max, n)
    local img_size  = F_mag:size()
    local img       = torch.Tensor(3, img_size[2], img_size[3]):fill(0)

    img[{ {1},{},{} }]   = (F_dir+math.pi):div(2*math.pi)
    img[{ {2},{},{} }]   = F_mag:div(F_mag:size(3)*0.5):clamp(0, 1)
    img[{ {3},{},{} }]:fill(1) 
    img[{ {2,3},{},{} }] = torch.cmin(torch.cmax(img[{ {2,3},{},{} }], 0), 1)
    img                  = image.hsv2rgb(img)
    img[{ {1},{},{} }]   = img[{ {1},{},{} }]:cmul(F_val)
    img[{ {2},{},{} }]   = img[{ {2},{},{} }]:cmul(F_val)
    img[{ {3},{},{} }]   = img[{ {3},{},{} }]:cmul(F_val)
    return img
end

return flow_utils
