from __future__ import absolute_import

import os
import numpy as np
from emu.nnadapter import NNAdapter
from keras.models import model_from_json, model_from_yaml, Model, Sequential
from keras import applications
from keras import backend
from collections import OrderedDict
from emu.docutil import doc_inherit


imagenet_mean = np.array([103.939, 116.779, 123.68])
imagenet_std = np.array([1., 1., 1.])


class KerasAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Keras models.
    An installation of Keras is required.
    """

    def __init__(self, model_cfg, model_weights, mean, std, inputsize, keep_outputs=None, use_gpu=False):
        """
        Initializes the adapter with a pretrained model from a filepath or keras-model identifier
        (see https://keras.io/applications/#available-models).

        Parameters
        ----------
        model_cfg : String or dict or list
            Model configuration defining the architecture.
            Available options:
             - Use a model identifier to define the model configuration (e.g. 'ResNet50', 'VGG19').
             - Use a file path to a yaml or json formatted file to define the model configuration.
             - Use a dict to define the model configuration (e.g. as from keras_model.get_config())
             - Use a list to define the model configuration of a Sequential model (e.g. as from keras_model.get_config())
        model_weights : String
            Available options:
             - File path to a HDF5 save file containing all weights
             - Identifier of data set (e.g. 'imagenet') if model identifier is used for model_cfg
        mean : ndarray or String
            Mean definition via array or filepath to .npy tensor.
        std : ndarray or String
            Standard deviation definition via array or filepath to .npy tensor.
        inputsize : tuple or list
            Target input data dimensionality of format: (height, width, channels).
            Used for rescaling given data in preprocess step.
        keep_outputs : list, tuple or set
            List of layer identifier strings to keep during a feed forward call to enable later access via
            get_layeroutput().
            By default no layer outputs but the last are kept.
            Consolidate get_layers() to identify layers.
        use_gpu : bool
            Flag to enable gpu use. Default: False
        """
        self.base_model = self._load_model_config(model_cfg, model_weights)

        if os.path.exists(model_weights):
            self.base_model.load_weights(model_weights)

        cfg = self.base_model.get_config()
        self.layers = OrderedDict()
        for layer in cfg['layers']:
            self.layers[layer['name']] = layer['class_name']

        if keep_outputs is None:
            self.keep_outputs = []
        else:
            self.keep_outputs = keep_outputs

        self.output_map = OrderedDict()
        for name, _ in self.layers.items():
            if name in self.keep_outputs:
                self.output_map[name] = len(self.output_map)

        if self.layers.keys()[-1] not in self.output_map:
            self.output_map[self.layers.keys()[-1]] = len(self.output_map)

        self.model = Model(inputs=self.base_model.input,
                           outputs=[self.base_model.get_layer(name).output for name in self.output_map.keys()])

        self.mean = self._load_mean_std(mean)
        self.std = self._load_mean_std(std)

        self.inputsize = inputsize
        self.use_gpu = use_gpu

        self.nomean_warn = True
        self.nostd_warn = True

        data_format = backend.image_data_format()
        if data_format == 'channels_first':
            self.dimorder = 'chw'
        else:
            self.dimorder = 'hwc'

        self.blobs = []

    @staticmethod
    def _load_mean_std(handle):
        """
        Loads mean/std values from a .npy file or returns the identity if already a numpy array.
        Parameters
        ----------
        handle : Can be either a numpy array or a filepath as string

        Returns
        ----------
        mean/std : Numpy array expressing mean/std
        """
        if type(handle) == str:
            if handle.endswith('.npy'):
                return np.load(handle)
            else:
                raise ValueError('Unknown file format. Known formats: .npy')
        elif type(handle) == np.ndarray:
            return handle
        elif handle is not None:
            raise ValueError('Unknown format. Expected .npy file or numpy array.')

    @staticmethod
    def _load_model_config(model_cfg, model_weights):
        if type(model_cfg) == str:
            if not os.path.exists(model_cfg):
                try:
                    class_ = getattr(applications, model_cfg)
                    return class_(weights=model_weights)
                except AttributeError:
                    available_mdls = [attr for attr in dir(applications) if callable(getattr(applications, attr))]
                    raise ValueError('Could not load pretrained model with key {}. '
                                     'Available models: {}'.format(model_cfg, ', '.join(available_mdls)))

            with open(model_cfg, 'r') as fileh:
                try:
                    return model_from_json(fileh)
                except ValueError:
                    pass

                try:
                    return model_from_yaml(fileh)
                except ValueError:
                    pass

            raise ValueError('Could not load model from configuration file {}. '
                             'Make sure the path is correct and the file format is yaml or json.'.format(model_cfg))
        elif type(model_cfg) == dict:
            return Model.from_config(model_cfg)
        elif type(model_cfg) == list:
            return Sequential.from_config(model_cfg)

        raise ValueError('Could not load model from configuration object of type {}.'.format(type(model_cfg)))

    @doc_inherit
    def get_layeroutput(self, layer):
        assert len(self.blobs) > 0, 'Forward has not been called. Layer outputs are not ready.'

        if layer not in self.output_map:
            if layer in self.get_layers():
                raise ValueError('Layer with id {} does exist in the architecture '
                                 'but has not been cached due to the output filter: [{}]'.format(
                                    layer,
                                    ','.join(self.keep_outputs),
                                 ))
            else:
                raise ValueError('Layer with id {} does not exist.'.format(layer))

        return self.blobs[self.output_map[layer]]

    @doc_inherit
    def get_layerparams(self, layer):
        params = self.model.get_layer(layer).get_weights()
        if len(params) == 1:
            return tuple(params[0], None)
        else:
            return tuple(params)

    @doc_inherit
    def set_weights(self, layer, weights):
        L = self.model.get_layer(layer)
        _, bias = L.get_weights()
        L.set_weights((weights, bias))

    @doc_inherit
    def set_bias(self, layer, bias):
        L = self.model.get_layer(layer)
        weights, _ = L.get_weights()
        L.set_weights((weights, bias))

    @doc_inherit
    def get_layers(self):
        return self.layers

    @doc_inherit
    def model_description(self):
        return self.model.name + '\n\t' + \
                                 '\n\t'.join([': '.join(entry) for entry in self.layers.items()])

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
        return NNAdapter.preprocess(listofimages, self.inputsize,
                                    self.mean, self.std,
                                    self.dimorder, channelorder='bgr', scale=255.0)

    @doc_inherit
    def forward(self, input):
        outputs = self.model.predict(input, batch_size=input.shape[0])
        self.blobs = outputs

        if type(outputs) is list:
            return outputs[-1]
        else:
            return outputs
