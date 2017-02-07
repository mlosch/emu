import os
import re
from collections import OrderedDict
import pkg_resources

import PyTorch
import PyTorchAug
import PyTorchHelpers
import numpy as np

from nnadapter import NNAdapter
import image


class TorchAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Torch models.
    An installation of Torch and pytorch is required.
    """

    def __init__(self, model_fp, required_modules, mean, std, inputsize):
        # Load the Lua class that does the actual computation
        lua_file_abs = pkg_resources.resource_filename(__name__, 'backend/torchadapter.lua')
        lua_file_rel = os.path.relpath(lua_file_abs, os.getcwd())

        adapterclass = PyTorchHelpers.load_lua_class(lua_file_rel, 'TorchAdapter')
        self.adapter = adapterclass()

        # Load necessary torch modules to run the model
        for module in required_modules:
            print('Loading torch module {}'.format(module))
            self.adapter.require(module)

        print('Loading model from {}'.format(model_fp))
        self.adapter.loadfrom(model_fp)

        # Load/set mean and std
        self.mean = TorchAdapter.load_mean_std(mean)
        self.std = TorchAdapter.load_mean_std(std)

        self.nomean_warn = True
        self.nostd_warn = True

        self.ready = False
        self.inputsize = inputsize

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
        if type(handle) == str and handle.endswith('.t7'):
            return PyTorchAug.load(handle).asNumpyTensor()
        elif type(handle) == np.ndarray:
            return handle

    def model_description(self):
        self.adapter.modeldescription()

    @staticmethod
    def _natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s[0])]

    def get_layers(self):
        unordered = self.adapter.getlayers()
        ordered = OrderedDict(sorted(unordered.items(), key=TorchAdapter._natural_sort_key))
        return ordered

    def set_weights(self, layerpath, weights):
        layerids = layerpath.split('.')
        layerids = [int(layer) for layer in layerids]

        torch_weights = PyTorch.asFloatTensor(weights)
        retval = self.adapter.set_weights(layerids, torch_weights)
        if retval == -1:
            raise ValueError('Layer with id {} does not exist.'.format(layerpath))
        elif retval == -2:
            raise ValueError('Layer with id {} does not hold any weights.'.format(layerpath))
        elif retval == -3:
            raise ValueError('Dimensions of mask and weights do not match for layer {}.'.format(layerpath))

    def set_bias(self, layerpath, bias):
        layerids = layerpath.split('.')
        layerids = [int(layer) for layer in layerids]

        torch_bias = PyTorch.asFloatTensor(bias)
        retval = self.adapter.set_bias(layerids, torch_bias)
        if retval == -1:
            raise ValueError('Layer with id {} does not exist.'.format(layerpath))
        elif retval == -2:
            raise ValueError('Layer with id {} does not hold any bias.'.format(layerpath))
        elif retval == -3:
            raise ValueError('Dimensions of mask and bias do not match for layer {}.'.format(layerpath))

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
        layerids = layerpath.split('.')
        layerids = [int(layer) for layer in layerids]

        torch_tuple = self.adapter.getlayerparams(layerids)

        if torch_tuple is not None and len(torch_tuple) == 0:
            raise ValueError('Layer with id {} does not contain any weights or bias.')
        if not torch_tuple:
            raise ValueError('Layer with id {} does not exist.'.format(layerpath))

        np_tuple = (torch_tuple['weight'].asNumpyTensor(), torch_tuple['bias'].asNumpyTensor() if 'bias' in torch_tuple else None)
        return np_tuple

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

        layerids = layerpath.split('.')
        layerids = [int(layer) for layer in layerids]

        out = self.adapter.getlayeroutput(layerids)

        if out is None:
            raise ValueError('Layer with id {} does not exist.'.format(layerpath))

        if type(out) is dict:
            clean_out = []
            for k, v in out.items():
                clean_out.append(v.asNumpyTensor())
            out = clean_out
        else:
            out = out.asNumpyTensor()

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

        input_tensor = PyTorch.asFloatTensor(input)

        # forward
        out = self.adapter.forward(input_tensor, True)
        if type(out) is dict:
            clean_out = []
            for k, v in out.items():
                clean_out.append(v.asNumpyTensor())
            out = clean_out
        else:
            out = out.asNumpyTensor()
        self.ready = True

        return out
