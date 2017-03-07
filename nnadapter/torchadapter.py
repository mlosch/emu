import re
import os
from collections import OrderedDict
from functools import partial

import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from .backend.torchlegacy import load_legacy_model, LambdaBase
import numpy as np

from nnadapter import NNAdapter
import image


class TorchAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Torch7/pytorch models.
    An installation of pytorch is required.
    """

    def __init__(self, model_fp, mean, std, inputsize, use_gpu=False, output_filter=[]):
        """
        Initializes the adapter with a pretrained model from a filepath or pytorch-model identifier
        (see https://github.com/pytorch/vision#models).

        Parameters
        ----------
        model_fp : String
            Filepath or model identifier.
        mean : ndarray or String
            Mean definition via array or filepath to .t7 torch tensor.
        std : ndarray or String
            Standard deviation definition via array or filepath to .t7 torch tensor.
        inputsize : tuple or list
            Target input data dimensionality of format: (channels, height, width).
            Used for rescaling given data in preprocess step.
        use_gpu : bool
            Flag to enable gpu use. Default: False
        output_filter : list, tuple or set
            List of layer identifier strings to disable the caching of specific layer outputs.
            Matches any location within strings and is case sensitive.
            This may be used to reduce the memory footprint during a forward pass while caching layer outputs.
            By default, this filter is empty so that every layer output is cached.
        """
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
        self.output_filter = output_filter
        self._register_forward_hooks(self.model)

        # Load/set mean and std
        self.mean = TorchAdapter.load_mean_std(mean)
        self.std = TorchAdapter.load_mean_std(std)

        self.nomean_warn = True
        self.nostd_warn = True

        self.ready = False
        self.inputsize = inputsize

        self.use_gpu = use_gpu

    def _register_forward_hooks(self, module, trace=[]):
        """
        Recursively registers the _nn_forward_hook method with each module
        while assigning an appropriate layer-path to each module
        """
        for key, mod in module._modules.items():
            trace.append(key)
            if key not in self.output_filter:
                mod.register_forward_hook(partial(self._nn_forward_hook, name='.'.join(trace)))
            self._register_forward_hooks(mod, trace)
            trace.pop()

    def _nn_forward_hook(self, module, input, output, name=''):
        if type(output) is list:
            self.blobs[name] = [o.data.clone() for o in output]
        else:
            self.blobs[name] = output.data.clone()

    @staticmethod
    def load_mean_std(handle):
        """
        Loads mean/std values from a .t7 file or returns the identity if already a numpy array.
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
            else:
                return torch.load(handle).numpy()
        elif type(handle) == np.ndarray:
            return handle

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

    def get_layers(self):
        layers = OrderedDict()
        self._get_layers(self.model, layers)
        return layers

    def _get_param(self, layerpath, keysuffix):
        key = layerpath + '.' + keysuffix
        if key not in self.state_dict:
            raise ValueError('Layer with id {} does not exist or does not hold any {}.'.format(layerpath, keysuffix))
        return self.state_dict[key]

    def _set_param(self, layerpath, keysuffix, values):
        key = layerpath + '.' + keysuffix
        if values.shape != self.state_dict[key].size():
            raise ValueError('Dimensions do not match for layer {}.'.format(layerpath))

        layer_values = self._get_param(layerpath, keysuffix)
        torch_values = torch.from_numpy(values)
        layer_values.copy_(torch_values)

    def set_weights(self, layerpath, weights):
        self._set_param(layerpath, 'weight', weights)

    def set_bias(self, layerpath, bias):
        self._set_param(layerpath, 'bias', bias)

    def get_layerparams(self, layerpath):
        """
        Return a copy of the parameters (weights, bias) of a layer.

        Parameters
        ----------
        layerpath : String
            Expected format: (%d.)*%d, e.g. 11.3.2 or 1
            specifying the location of the layer within the torch model.
            To see possible locations of a model, call `model_description`.

        Returns
        -------
        (weights, bias) : Tuple of ndarrays
        """
        weight_key = layerpath + '.weight'
        bias_key = layerpath + '.bias'
        weights = self._get_param(layerpath, 'weight') if weight_key in self.state_dict else None
        bias = self._get_param(layerpath, 'bias') if bias_key in self.state_dict else None

        np_weights = weights.cpu().numpy() if weights is not None else None
        np_bias = bias.cpu().numpy() if bias is not None else None

        return np_weights, np_bias

    def get_layeroutput(self, layerpath):
        """
        Get the output of a specific layer.
        forward(...) has to be called in advance.

        Parameters
        ----------
        layerpath : String, Layer identification
            Expected format: (%d.)*%d, e.g. 11.3.2 or 1
            specifying the location of the layer within the torch model.
            To see possible locations of a model, call `model_description`.

        Returns
        -------
        output : ndarray
            Numpy tensor of output values.
        """
        assert self.ready, 'Forward has not been called. Layer outputs are not ready.'

        if layerpath not in self.blobs:
            if layerpath in self.get_layers():
                raise ValueError('Layer with id {} does exist in the architecture '
                                 'but has not been cached due to the output filter: [{}]'.format(
                                    layerpath,
                                    ','.join(self.output_filter),
                                 ))
            else:
                raise ValueError('Layer with id {} does not exist.'.format(layerpath))

        out = self.blobs[layerpath]

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

        # Load data in first step from list
        data = np.zeros((len(listofimages), self.inputsize[0], self.inputsize[1], self.inputsize[2]), dtype=np.float32)

        for i, h in enumerate(listofimages):
            if type(h) == str:
                im = image.read(h)
            elif type(h) == np.ndarray:
                im = h

            im = image.resize(im, self.inputsize[1:])

            if self.mean is not None and self.mean.ndim == 1:
                im -= self.mean
            if self.std is not None and self.std.ndim == 1:
                im /= self.std
            im = im.transpose(2, 0, 1)  # resulting order is: channels x height x width
            if self.mean is not None and self.mean.ndim == 3:
                im -= self.mean
            if self.std is not None and self.std.ndim == 3:
                im /= self.std

            data[i] = im

        return data

    def forward(self, input):
        """
        Forward a batch of images through the neural network.

        Parameters
        ----------
        input : ndarray
            Batch of preprocessed images with 4 dimensions: (Batch, Channels, Height, Width).

        Returns
        -------
        output : ndarray
            Output of final network layer.
        """

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

    def _traverse_model(self, pathtolayer):
        m = self.model
        if type(pathtolayer) is list:
            for layer in pathtolayer:
                if len(m._modules) == 0:
                    raise KeyError('Layer {} does not exist .{} does not contain the requested sub-node.'.format(
                        pathtolayer, type(m)
                    ))
                m = m.modules[layer]
            return m
        else:
            return m.modules[pathtolayer]
