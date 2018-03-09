local utils = {}

-- Simple Function to Manipulate tables in lua
function utils.dictLength(T)
   local count = 0
      for _ in pairs(T) do count = count + 1 end
   return count
end

function utils.dictReset(t)
    for k,v in pairs(t) do
        t[k] = 0
    end
    return t
end

function utils.dictAddKeys(dict, t)
    for k,v in pairs(t) do
        dict[k] = {}
    end
    return dict
end

function utils.insertSubDicts(dict, sub_dict)
    for k, v in pairs(sub_dict) do
        if dict[k] == nil then
            dict = utils.dictAddKeys(dict, sub_dict)
        end
        table.insert(dict[k], v)
    end
    return dict 
end

function utils.dictsAdd(dict1, dict2)
    for k,v in pairs(dict2) do
        if dict1[k] == nil then dict1[k] = 0 end
        dict1[k] = dict1[k] + v
    end
    return dict1
end

function utils.dictDivide(t, n)
    local tab = {}
    for k,v in pairs(t) do
        tab[k] = v / n
    end
    return tab
end

function utils.dictOfDictAverage(dict_of_dict)
    local dict = {}
    local n = 0
    for i,d in pairs(dict_of_dict) do
        for k,v in pairs(d) do
            if dict[k] == nil then dict[k] = 0 end
            dict[k] = dict[k] + v
        end
        n = n + 1
    end
    for k,v in pairs(dict) do 
        dict[k] = dict[k] / n
    end
    return dict
end

return utils

