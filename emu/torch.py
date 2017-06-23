from __future__ import absolute_import

import re
import os
from collections import OrderedDict
from functools import partial

import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from .backend.torchlegacy import load_legacy_model, LambdaBase
import numpy as np

from emu.nnadapter import NNAdapter
from emu.docutil import doc_inherit


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class TorchAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Torch7/pytorch models.
    An installation of pytorch is required.
    """

    def __init__(self, model_fp, mean, std, inputsize, keep_outputs=None, use_gpu=False):
        """
        Initializes the adapter with a pretrained model from a filepath or pytorch-model identifier
        (see https://github.com/pytorch/vision#models).

        Parameters
        ----------
        model_fp : String
            Filepath or model identifier.
        mean : ndarray or String
            Mean definition via array or filepath to .t7 torch or .npy tensor.
        std : ndarray or String
            Standard deviation definition via array or filepath to .t7 torch or .npy tensor.
        inputsize : tuple or list
            Target input data dimensionality of format: (channels, height, width).
            Used for rescaling given data in preprocess step.
        use_gpu : bool
            Flag to enable gpu use. Default: False
        keep_outputs : list, tuple or set
            List of layer identifier strings to keep during a feed forward call to enable later access via
            get_layeroutput().
            By default no layer outputs but the last are kept.
            Consolidate get_layers() to identify layers.
        """
        # self.model = self._load_model_config(model_fp)
        if '.' not in os.path.basename(model_fp):
            import torchvision.models as models
            if model_fp not in models.__dict__:
                raise KeyError('Model {} does not exist in pytorchs model zoo.'.format(model_fp))
            print('Loading model {} from pytorch model zoo'.format(model_fp))
            self.model = models.__dict__[model_fp](pretrained=True)
        else:
            print('Loading model from {}'.format(model_fp))
            if model_fp.endswith('.t7'):
                self.model = load_legacy_model(model_fp)
            else:
                self.model = torch.load(model_fp)
        self.model.train(False)

        if use_gpu:
            self.model.cuda()
        else:
            self.model.float()

        self.blobs = {}
        self.state_dict = self.model.state_dict()

        # register forward hooks with model
        if keep_outputs is None:
            self.keep_outputs = []
        else:
            self.keep_outputs = keep_outputs
        self._register_forward_hooks(self.model)

        # Load/set mean and std
        self.mean = TorchAdapter._load_mean_std(mean)
        self.std = TorchAdapter._load_mean_std(std)

        self.nomean_warn = True
        self.nostd_warn = True

        self.ready = False
        self.inputsize = tuple(inputsize)

        self.use_gpu = use_gpu

    def _register_forward_hooks(self, module, trace=[]):
        """
        Recursively registers the _nn_forward_hook method with each module
        while assigning an appropriate layer-path to each module
        """
        for key, mod in module._modules.items():
            trace.append(key)
            if key in self.keep_outputs:
                mod.register_forward_hook(partial(self._nn_forward_hook, name='.'.join(trace)))
            self._register_forward_hooks(mod, trace)
            trace.pop()

    def _nn_forward_hook(self, module, input, output, name=''):
        if type(output) is list:
            self.blobs[name] = [o.data.clone() for o in output]
        else:
            self.blobs[name] = output.data.clone()

    # @staticmethod
    # def _load_model_config(model_def):
    #     if isinstance(model_def, torch.nn.Module):
    #
    #     elif '.' not in os.path.basename(model_def):
    #         import torchvision.models as models
    #         if model_def not in models.__dict__:
    #             raise KeyError('Model {} does not exist in pytorchs model zoo.'.format(model_def))
    #         print('Loading model {} from pytorch model zoo'.format(model_def))
    #         return models.__dict__[model_def](pretrained=True)
    #     else:
    #         print('Loading model from {}'.format(model_def))
    #         if model_def.endswith('.t7'):
    #             return load_legacy_model(model_def)
    #         else:
    #             return torch.load(model_def)
    #
    #
    #     if type(model_cfg) == str:
    #         if not os.path.exists(model_cfg):
    #             try:
    #                 class_ = getattr(applications, model_cfg)
    #                 return class_(weights=model_weights)
    #             except AttributeError:
    #                 available_mdls = [attr for attr in dir(applications) if callable(getattr(applications, attr))]
    #                 raise ValueError('Could not load pretrained model with key {}. '
    #                                  'Available models: {}'.format(model_cfg, ', '.join(available_mdls)))
    #
    #         with open(model_cfg, 'r') as fileh:
    #             try:
    #                 return model_from_json(fileh)
    #             except ValueError:
    #                 pass
    #
    #             try:
    #                 return model_from_yaml(fileh)
    #             except ValueError:
    #                 pass
    #
    #         raise ValueError('Could not load model from configuration file {}. '
    #                          'Make sure the path is correct and the file format is yaml or json.'.format(model_cfg))
    #     elif type(model_cfg) == dict:
    #         return Model.from_config(model_cfg)
    #     elif type(model_cfg) == list:
    #         return Sequential.from_config(model_cfg)
    #
    #     raise ValueError('Could not load model from configuration object of type {}.'.format(type(model_cfg)))

    @staticmethod
    def _load_mean_std(handle):
        """
        Loads mean/std values from a .t7/.npy file or returns the identity if already a numpy array.
        Parameters
        ----------
        handle : Can be either a numpy array or a filepath as string

        Returns
        ----------
        mean/std : Numpy array expressing mean/std
        """
        if type(handle) == str:
            if handle.endswith('.t7'):
                return load_lua(handle).numpy()
            elif handle.endswith('.npy'):
                return np.load(handle)
            else:
                return torch.load(handle).numpy()
        elif type(handle) == np.ndarray:
            return handle

    @doc_inherit
    def model_description(self):
        return str(self.model)

    @staticmethod
    def _natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s[0])]

    def _get_layers(self, module, dictionary, trace=[]):
        for key, mod in module._modules.items():
            trace.append(key)
            """
            str(mod) returns too complicated descriptions for modules that contain children.
            We limit the description to its name.
            LambdaBase may be used for legacy torch7 models.
            """
            if isinstance(mod, LambdaBase):
                desc = '{} ({})'.format(mod, mod.__class__.__name__)
            elif len(mod._modules) > 0:
                desc = mod.__class__.__name__
            else:
                desc = str(mod)
            dictionary['.'.join(trace)] = desc
            self._get_layers(mod, dictionary, trace)
            trace.pop()

    @doc_inherit
    def get_layers(self):
        layers = OrderedDict()
        self._get_layers(self.model, layers)
        return layers

    def _get_param(self, layer, keysuffix):
        key = layer + '.' + keysuffix
        if key not in self.state_dict:
            raise ValueError('Layer with id {} does not exist or does not hold any {}.'.format(layer, keysuffix))
        return self.state_dict[key]

    def _set_param(self, layer, keysuffix, values):
        key = layer + '.' + keysuffix
        if values.shape != self.state_dict[key].size():
            raise ValueError('Dimensions do not match for layer {}.'.format(layer))

        layer_values = self._get_param(layer, keysuffix)
        torch_values = torch.from_numpy(values)
        layer_values.copy_(torch_values)

    @doc_inherit
    def set_weights(self, layer, weights):
        self._set_param(layer, 'weight', weights)

    @doc_inherit
    def set_bias(self, layer, bias):
        self._set_param(layer, 'bias', bias)

    @doc_inherit
    def get_layerparams(self, layer):
        weight_key = layer + '.weight'
        bias_key = layer + '.bias'
        weights = self._get_param(layer, 'weight') if weight_key in self.state_dict else None
        bias = self._get_param(layer, 'bias') if bias_key in self.state_dict else None

        np_weights = weights.cpu().numpy() if weights is not None else None
        np_bias = bias.cpu().numpy() if bias is not None else None

        return np_weights, np_bias

    @doc_inherit
    def get_layeroutput(self, layer):
        assert self.ready, 'Forward has not been called. Layer outputs are not ready.'

        if layer not in self.blobs:
            if layer in self.get_layers():
                raise ValueError('Layer with id {} does exist in the architecture '
                                 'but has not been cached due to the output filter: [{}]'.format(
                                    layer,
                                    ','.join(self.keep_outputs),
                                 ))
            else:
                raise ValueError('Layer with id {} does not exist.'.format(layer))

        out = self.blobs[layer]

        if type(out) is list:
            clean_out = []
            for v in out:
                clean_out.append(v.cpu().numpy())
            out = clean_out
        else:
            out = out.cpu().numpy()

        return out

    def preprocess(self, listofimages):
        """
        Preprocess a list of images to be used with the neural network.

        Parameters
        ----------
        listofimages : List of strings or list of ndarrays, shape (Height, Width, Channels)
            The list may contain image filepaths and image ndarrays.
            For ndarrays, the shape (Height, Width, Channels) has to conform with the input size defined at
            object construction.

        Returns
        -------
        output : ndarray
            Preprocessed batch of images.
        """
        if self.mean is None and self.nomean_warn:
            print('Warning: No mean specified.')
            self.nomean_warn = False

        if self.std is None and self.nostd_warn:
            print('Warning: No standard deviation specified.')
            self.nostd_warn = False

        return NNAdapter.preprocess(listofimages, self.inputsize, self.mean, self.std)

    @doc_inherit
    def forward(self, input):
        input_torch = torch.from_numpy(input)
        if self.use_gpu:
            input_torch = input_torch.cuda()
        else:
            input_torch = input_torch.float()

        input_var = Variable(input_torch)

        # forward
        out = self.model.forward(input_var)

        if type(out) is list:
            clean_out = []
            for v in out:
                clean_out.append(v.data.cpu().numpy())
            out = clean_out
        else:
            out = out.data.cpu().numpy()
        self.ready = True

        return out

    @staticmethod
    def _find(module, layerid, trace=[]):
        if '.'.join(trace) == layerid:
            return module

        for key, mod in module._modules.items():
            trace.append(key)
            if layerid.startswith('.'.join(trace)):
                result = TorchAdapter._find(mod, layerid, trace)
                if result is not None:  # multiple layer ids may have the same prefix
                    return result
            trace.pop()

        return None

    @staticmethod
    def _gkern(kernlen=21, nsig=3):
        import scipy.stats as st
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / kernlen
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def visualize(self, input, layer, unitidx, lr=1.0, iters=500, eps=1e-6):
        if isinstance(unitidx, list) and input is not None and input.shape[0] != len(unitidx):
            raise ValueError('List of unit indices cannot be passed together with an input batch of unequal samples. '
                             'Pass None for input alternatively.')
        if input is None:
            if isinstance(unitidx, list):
                sz = (len(unitidx),) + self.inputsize
            else:
                sz = (1,) + self.inputsize
            input_torch = torch.randn(*sz)
        else:
            input_torch = torch.from_numpy(input)

        # find module
        module = TorchAdapter._find(self.model, layer, trace=[])
        if module is None:
            raise RuntimeError('Could not find layer %s' % layer)

        gamma = 0.2
        blur_every = 4
        percentile = 0.0001

        # Gaussian blur kernel for smoothing visualization
        gkernsz = 7
        gaussian_krnl = torch.from_numpy(TorchAdapter._gkern(gkernsz, 3))
        # gaussian_krnl = torch.Tensor([
        #     [0.0509,  0.1238,  0.0509],
        #     [0.1238,  0.3012,  0.1238],
        #     [0.0509,  0.1238,  0.0509],
        # ])
        blur = torch.nn.Conv2d(3, 3, gkernsz, padding=gkernsz//2, bias=False)
        blur.weight.data[0, 0, :, :].copy_(gaussian_krnl)
        blur.weight.data[1, 1, :, :].copy_(gaussian_krnl)
        blur.weight.data[2, 2, :, :].copy_(gaussian_krnl)

        # Transfer data to gpu
        if self.use_gpu:
            input_torch = input_torch.cuda()
            blur = blur.cuda()
        else:
            input_torch = input_torch.float()

        input = Variable(input_torch, requires_grad=True)

        # determine target tensor for unit
        self.model.forward(input)

        target = Variable(self.blobs[layer])
        target.data.zero_()

        if target.dim() == 2:
            if isinstance(unitidx, list):
                for i, unit in enumerate(unitidx):
                    target.data[i, unit] = 1.0
            else:
                target.data[:, unitidx] = 1.0
        else:
            if isinstance(unitidx, list):
                for i, unit in enumerate(unitidx):
                    target.data[i, unit, target.size(2) // 2, target.size(3) // 2] = 1.0
            else:
                target.data[:, unitidx, target.size(2) // 2, target.size(3) // 2] = 1.0

        loss = [None]

        def criterion_latch_on(module, input, output):
            loss[0] = output
        hook_handle = module.register_forward_hook(criterion_latch_on)

        for i in range(iters):

            self.model.forward(input)
            loss[0].backward(target.data)

            # sum = 0
            # for k in range(3):
            #     sum += torch.dot(input.data[0, 0], input.grad.data[0, 0])
            # lower_bound = torch.abs(sum) * percentile

            input.data.add_(lr, input.grad.data)
            input.data.mul_(1.0-gamma)
            input.grad.data.zero_()

            if i % blur_every == 0:
                input.data.copy_(blur.forward(input).data)

            # if abs(loss[0].data[0] - last_loss) < eps:
            #     break
            # last_loss = loss[0].data[0]

        hook_handle.remove()

        return input.data.cpu().numpy()

