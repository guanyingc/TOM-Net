require 'paths'
local str_utils = {}
-- Simple string utils
function str_utils.splitext(fname)
    local ext = paths.extname(fname)
    if ext == nil then
        return ''
    end
    local start = fname:find(ext)
    ext = fname:sub(start-1, -1)
    local name = fname:sub(1, start-2)
    return name, ext
end

function str_utils.time_left(startTime, nEpoches, batches, curEpoch, curIter)
    local curTime = os.time()
    local timeSofar = (curTime - startTime) / 3600.0
    local total_step = nEpoches * batches
    local cur_step = (curEpoch - 1) * batches + curIter
    local time_left = timeSofar * (total_step / cur_step - 1)
    local s = string.format('Time elapsed: %.2fh | Time left %.2fh', timeSofar, time_left)
    return s
end

function str_utils.time_used(startTime)
    local curTime = os.time()
    local time_used = (curTime - startTime) / 3600.0
    local s = string.format('Time elapsed: %.2fh', time_used)
    return s
end

function str_utils.add_time(time, timer)
   if cutorch then cutorch.synchronize() end
   time = time + timer:time().real
   timer:reset()
   return time
end

function str_utils.build_loss_string(losses, no_total)
    local s = '\t'
    local total_loss = 0
    for i,v in pairs(losses) do
        s = s .. string.format('%s: %.5f, ', i, v)
        total_loss = total_loss + v
    end
    if not no_total then 
        s = s .. string.format(' [Total Loss: %.5f]', total_loss)
    end
    return s
end

function str_utils.build_time_string(times, no_total)
    local s = '\t'
    local total_time = 0
    for i,v in pairs(times) do
        s = s .. string.format('%s: %.3fs, ', i, v)
        total_time = total_time + v
    end
    if not no_total then 
        s = s .. string.format(' [Total Time: %.3f]', total_time)
    end
    return s
end

function str_utils.dict_to_string(t)
    local s = '\t'
    for i,v in pairs(t) do
        s = s .. string.format('%s: %.5f, ', i, v)
    end
    return s
end

function str_utils.getDateTime()
   local date = os.date():split(' ')
   local day; local week; local mon; local time;
   if #date == 6 then 
      day = date[3] .. date[4]
      week = date[1]
      mon = date[2]
      time = date[5]
   else
       day = date[3]
       week = date[1]
       mon = date[2]
       time = date[4]
   end
   date = string.format('%s-%s', mon, day) 
   return date, time
end

return str_utils
