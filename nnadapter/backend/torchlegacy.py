"""
Thanks to clcarwin
https://github.com/clcarwin/convert_torch_to_pytorch
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.serialization import load_lua


class LambdaBase(nn.Sequential):
    def __init__(self, name, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
        self.name = name

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

    def __str__(self):
        return self.name


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return map(self.lambda_func, self.forward_prepare(input))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func, self.forward_prepare(input))


def copy_param(m, n):
    if m.weight is not None: n.weight.data.copy_(m.weight)
    if m.bias is not None: n.bias.data.copy_(m.bias)
    if hasattr(n, 'running_mean'): n.running_mean.copy_(m.running_mean)
    if hasattr(n, 'running_var'): n.running_var.copy_(m.running_var)


def add_submodule(seq, *args):
    for n in args:
        seq.add_module(str(len(seq._modules)), n)


def lua_recursive_model(module, seq):
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.', '')
            m = m._obj

        if name == 'SpatialConvolution':
            if not hasattr(m, 'groups'): m.groups = 1
            n = nn.Conv2d(m.nInputPlane, m.nOutputPlane, (m.kW, m.kH), (m.dW, m.dH), (m.padW, m.padH), 1, m.groups,
                          bias=(m.bias is not None))
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'SpatialBatchNormalization':
            n = nn.BatchNorm2d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'ReLU' or name == 'Threshold':
            n = nn.ReLU()
            add_submodule(seq, n)
        elif name == 'SpatialMaxPooling':
            n = nn.MaxPool2d((m.kW, m.kH), (m.dW, m.dH), (m.padW, m.padH), ceil_mode=m.ceil_mode)
            add_submodule(seq, n)
        elif name == 'SpatialAveragePooling':
            n = nn.AvgPool2d((m.kW, m.kH), (m.dW, m.dH), (m.padW, m.padH), ceil_mode=m.ceil_mode)
            add_submodule(seq, n)
        elif name == 'SpatialUpSamplingNearest':
            n = nn.UpsamplingNearest2d(scale_factor=m.scale_factor)
            add_submodule(seq, n)
        elif name == 'View':
            n = Lambda(name, lambda x: x.view(x.size(0), -1))
            add_submodule(seq, n)
        elif name == 'Linear':
            # Linear in pytorch only accept 2D input
            #n1 = Lambda(name, lambda x: x.view(1, -1) if 1 == len(x.size()) else x)
            n2 = nn.Linear(m.weight.size(1), m.weight.size(0), bias=(m.bias is not None))
            copy_param(m, n2)
            #n = nn.Sequential(n1, n2)
            add_submodule(seq, n2)
        elif name == 'Dropout':
            m.inplace = False
            n = nn.Dropout(m.p)
            add_submodule(seq, n)
        elif name == 'SoftMax':
            n = nn.Softmax()
            add_submodule(seq, n)
        elif name == 'Identity':
            n = Lambda(name, lambda x: x)  # do nothing
            add_submodule(seq, n)
        elif name == 'SpatialFullConvolution':
            n = nn.ConvTranspose2d(m.nInputPlane, m.nOutputPlane, (m.kW, m.kH), (m.dW, m.dH), (m.padW, m.padH))
            add_submodule(seq, n)
        elif name == 'SpatialReplicationPadding':
            n = nn.ReplicationPad2d((m.pad_l, m.pad_r, m.pad_t, m.pad_b))
            add_submodule(seq, n)
        elif name == 'SpatialReflectionPadding':
            n = nn.ReflectionPad2d((m.pad_l, m.pad_r, m.pad_t, m.pad_b))
            add_submodule(seq, n)
        elif name == 'Copy':
            n = Lambda(name, lambda x: x)  # do nothing
            add_submodule(seq, n)
        elif name == 'Narrow':
            n = Lambda(name, lambda x, a=(m.dimension, m.index, m.length): x.narrow(*a))
            add_submodule(seq, n)
        elif name == 'SpatialCrossMapLRN':
            lrn = torch.legacy.nn.SpatialCrossMapLRN(m.size, m.alpha, m.beta, m.k)
            n = Lambda(name, lambda x, lrn=lrn: Variable(lrn.forward(x.data)))
            add_submodule(seq, n)
        elif name == 'Sequential':
            n = nn.Sequential()
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'ConcatTable':  # output is list
            n = LambdaMap(name, lambda x: x)
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'CAddTable':  # input is list
            n = LambdaReduce(lambda x, y: x + y)
            add_submodule(seq, n)
        elif name == 'Concat':
            dim = m.dimension
            n = LambdaReduce(name, lambda x, y, dim=dim: torch.cat((x, y), dim))
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'TorchObject':
            print('Not Implement', name, real._typename)
        else:
            print('Not Implement', name)


def load_legacy_model(t7_filename):
    model = load_lua(t7_filename, unknown_classes=True)
    if type(model).__name__ == 'hashable_uniq_dict':
        model = model.model
    model.gradInput = None

    n = nn.Sequential()
    lua_recursive_model(model, n)
    return n
