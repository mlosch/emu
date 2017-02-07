require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local torchadapter = torch.class('TorchAdapter')

function string.endswith(self, suffix)
    return self:reverse():find(string.reverse(suffix)) == 1
end

function torch.LongStorage.equals(self, storage)
    if #self ~= #storage then
        return false
    end
    for i=1,#self do
       if self[i] ~= storage[i] then
          return false
       end
    end
    return true
end

function torchadapter.require(self, mod)
    --if mod:reverse():find(string.reverse('.lua')) == 1 then
    if mod:endswith('.lua') then
        -- mod appears to be a lua script
        dofile(mod)
    else
        require(mod)
    end
end

function torchadapter.loadfrom(self, filepath)
    self.model = torch.load(filepath)
    -- set into evaluation mode
    self.model:evaluate()
end

function torchadapter.tocpu(tensor_or_table)
    if torch.type(tensor_or_table) == 'table' then
        for k, v in ipairs(tensor_or_table) do
            tensor_or_table[k] = v:float()
        end
    else
        tensor_or_table = tensor_or_table:float()
    end
    return tensor_or_table
end

function torchadapter.forward(self, batch, useGPU)
    if useGPU then
        batch = batch:cuda()
        self.model:cuda()
    else
        batch = batch:float()
        self.model:float()
    end

    self.useGPU = useGPU

    local out = self.model:forward(batch)

    return self.tocpu(out)
end

function torchadapter.modeldescription(self)
    print(self.model)
end

function torchadapter.addsublayers(self, dict, module, prefix)
    prefix = prefix or ''

    for k, m in ipairs(module.modules) do
        dict[prefix .. k] = torch.type(m)
        if m.modules then
            dict = self:addsublayers(dict, m, prefix .. k .. '.')
        end
    end

    return dict
end

function torchadapter.getlayers(self)
    local layers = self:addsublayers({}, self.model, '')
    return layers
end

function torchadapter._traversemodel(self, pathtolayer)
    local m = self.model
    if type(pathtolayer) == 'table' then
        for _, layer in ipairs(pathtolayer) do
            if m.modules == nil then
                return nil
            end
            m = m.modules[layer]
        end
        return m
    else
        return m.modules[pathtolayer]
    end
end

function torchadapter.getlayerparams(self, pathtolayer)
    local m = self:_traversemodel(pathtolayer)
    if m == nil then
        return nil
    end
    local tuple = {['weight'] = m.weight, ['bias'] = m.bias}
    for k, v in pairs(tuple) do
        tuple[k] = v:float()
    end
    return tuple
end

function torchadapter.set_weights(self, pathtolayer, weights)
    local m = self:_traversemodel(pathtolayer)
    if m == nil then
        return -1
    end
    if m.weight == nil then
        return -2
    end
    if not m.weight:size():equals(weights:size()) then
        return -2
    end

    if self.useGPU then
        weights = weights:cuda()
    end
    m.weight:set(weights)

    return 0
end

function torchadapter.set_bias(self, pathtolayer, bias)
    local m = self:_traversemodel(pathtolayer)
    if m == nil then
        return -1
    end
    if m.bias == nil then
        return -2
    end
    if not m.bias:size():equals(bias:size()) then
        return -3
    end

    if self.useGPU then
        bias = bias:cuda()
    end
    m.bias:set(bias)

    return 0
end

function torchadapter.getlayeroutput(self, pathtolayer)
    local m = self:_traversemodel(pathtolayer)
    if m == nil then
        return nil
    end
    if m.output == nil then
        print('Layer '.. str(pathtolayer) ..' does exist, but has no output data.')
        return nil
    end
    return self.tocpu(m.output)
end
