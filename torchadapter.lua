require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local torchadapter = torch.class('TorchAdapter')

function string.endswith(self, suffix)
    return self:reverse():find(string.reverse(suffix)) == 1
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

function torchadapter.forward(self, batch, useGPU)
    if useGPU then
        batch = batch:cuda()
        self.model:cuda()
    else
        batch = batch:float()
        self.model:float()
    end

    local out = self.model:forward(batch)

    return out:float()
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

function torchadapter.getlayeroutput(self, pathtolayer)
    local m = self:_traversemodel(pathtolayer)
    if m == nil then
        return nil
    end
    return m.output:float()
end
