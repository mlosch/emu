import re
from collections import OrderedDict

import PyTorch
import PyTorchAug
import PyTorchHelpers
import numpy as np

from nnadapter import NNAdapter


class TorchAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Torch models.
    An installation of Torch and pytorch is required.
    """
    def __init__(self, model_fp, required_modules, mean, std, inputsize):
        # Load the Lua class that does the actual computation
        adapterclass = PyTorchHelpers.load_lua_class('nnadapter/backend/torchadapter.lua', 'TorchAdapter')
        self.adapter = adapterclass()

        # Load necessary torch modules to run the model
        for module in required_modules:
            print('Loading torch module {}'.format(module))
            self.adapter.require(module)

        print('Loading model from {}'.format(model_fp))
        self.adapter.loadfrom(model_fp)

        # Load/set mean and std
        self.mean = TorchAdapter.load_mean_std(mean)
        if self.mean is None:
            print('Warning. No mean specified.')
            # raise ValueError('Unknown mean format. Expected .t7 file or numpy array.')
        self.std = TorchAdapter.load_mean_std(std)
        if self.std is None:
            print('Warning. No standard deviation specified.')
            # raise ValueError('Unknown std format. Expected .t7 file or numpy array.')

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

        np_tuple = (torch_tuple['weight'].asNumpyTensor(), torch_tuple['bias'].asNumpyTensor())
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

        torch_tensor = self.adapter.getlayeroutput(layerids)

        if not torch_tensor:
            raise ValueError('Layer with id {} does not exist.'.format(layerpath))
        np_tensor = torch_tensor.asNumpyTensor()
        return np_tensor

    def forward(self, input):
        """
        Forward a batch of images through the neural network.

        Parameters
        ----------
        input : List of strings or list of ndarrays, shape (Height, Width, Channels)
            The list may contain image filepaths and image ndarrays.
            For ndarrays, the shape (Height, Width, Channels) has to conform with the input size defined at
            object construction.

        Returns
        -------
        output : ndarray
            Output of final network layer.
        """

        # Load data in first step from list
        data = np.zeros((len(input), self.inputsize[0], self.inputsize[1], self.inputsize[2]), dtype=np.float32)

        for i, h in enumerate(input):
            if type(h) == str:
                im = nnadapter.image.read(h)
            elif type(h) == np.ndarray:
                im = h

            im = nnadapter.image.resize(im, self.inputsize[1:])

            if self.mean.ndim == 1:
                im -= self.mean
            if self.std.ndim == 1:
                im /= self.std
            im = im.transpose(2, 0, 1)  # resulting order is: channels x height x width
            if self.mean.ndim == 3:
                im -= self.mean
            if self.std.ndim == 3:
                im /= self.std

            data[i] = im

        input_tensor = PyTorch.asFloatTensor(data)

        # forward
        out = self.adapter.forward(input_tensor, True)
        out = out.asNumpyTensor()
        self.ready = True

        return out
