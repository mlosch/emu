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

function torchadapter.visualize(self, input, pathtolayer, target, mean, std)
    local m = nn.Sequential()

    for i=1,pathtolayer[3] do
        m:add(self.model:get(1):get(1):get(i))
    end

    local alpha = 1.0
    local gamma = 0.01
    local blur_every = 10
    local percentile = 0.3

    local gaussian_krnl = torch.Tensor({
        {0.0509,  0.1238,  0.0509},
        {0.1238,  0.3012,  0.1238},
        {0.0509,  0.1238,  0.0509}})
    local blur = nn.SpatialConvolution(3, 3, 3, 3, 1, 1, 1, 1)
    blur.bias:zero()
    blur.weight:zero()
    blur.weight[{1, 1, {}, {}}]:copy(gaussian_krnl)
    blur.weight[{2, 2, {}, {}}]:copy(gaussian_krnl)
    blur.weight[{3, 3, {}, {}}]:copy(gaussian_krnl)

    local criterion = nn.MSECriterion()
    if self.useGPU then
        criterion = criterion:cuda()
        input = input:cuda()
        target = target:cuda()
        blur = blur:cuda()
    end

    local err = 0
    for i=1,1000 do
        for k=1,3 do
            input[{1,k,{},{}}]:mul(std[k])
            input[{1,k,{},{}}]:add(mean[k])

            input[{1,k,{},{}}]:clamp(0.0, 1.0)

            input[{1,k,{},{}}]:add(-mean[k])
            input[{1,k,{},{}}]:mul(1 / std[k])
        end
        local output = m:forward(input)

        err = criterion:forward(output, target)
        --local da_dt = criterion:backward(output, target)
        local di_da = m:updateGradInput(input, target)

        input:add(alpha, di_da)
        input:mul(1-gamma)

        if i % blur_every == 0 then
            input:copy(blur:forward(input))
        end

        local i_min = input:min()
        local i_max = input:max()
        local lower_bound = i_min + (i_max - i_min) * percentile
        local mask = torch.abs(input):lt(lower_bound)
        input[mask] = 0

        --print(''.. i ..': '.. err)
    end
    print(err)

    return input:float()

end
