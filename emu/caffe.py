from __future__ import absolute_import

from emu.nnadapter import NNAdapter
from collections import OrderedDict
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from emu.docutil import doc_inherit


imagenet_mean = np.array([104.00698793, 116.66876762, 122.67891434])


class CaffeAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Caffe models.
    An installation of Caffe and pycaffe is required.
    """

    def __init__(self, prototxt, caffemodel, mean, use_gpu=False):
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        if use_gpu:
            caffe.set_mode_gpu()

        if type(mean) == str:
            if mean.endswith('.binaryproto'):
                self.mean = CaffeAdapter._load_binaryproto(mean)
            elif mean.endswith('.npy'):
                self.mean = np.load(mean)
            else:
                raise ValueError('Unknown mean file format. Known formats: .binaryproto, .npy')
        elif type(mean) == np.ndarray:
            self.mean = mean
        elif mean is not None:
            raise ValueError('Unknown mean format. Expected .binaryproto/.npy file or numpy array.')

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        if self.mean is not None:
            self.transformer.set_mean('data', self.mean.mean(1).mean(1))
        else:
            print('Warning. No mean specified.')
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        self.layer_types = self._load_layer_types(prototxt)

        self.ready = False

        self.use_gpu = use_gpu

    @staticmethod
    def _load_layer_types(prototxt):
        # Read prototxt with caffe protobuf definitions
        layers = caffe_pb2.NetParameter()
        with open(prototxt, 'r') as f:
            text_format.Merge(str(f.read()), layers)

        # Assign layer parameters to type dictionary
        types = OrderedDict()
        for i in range(len(layers.layer)):
            types[layers.layer[i].name] = layers.layer[i].type

        return types

    @staticmethod
    def _load_binaryproto(file):
        blob = caffe_pb2.BlobProto()
        data = open(file, 'rb').read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        return arr[0]

    @doc_inherit
    def model_description(self):
        string = ''
        for k, v in self.net.blobs.items():
            string += '{}: {}\n'.format(k, v)
        return string

    @doc_inherit
    def get_layers(self):
        return self.layer_types

    @doc_inherit
    def get_layerparams(self, layer):
        if layer not in self.net.params:
            return None
        return self.net.params[layer][0].data, self.net.params[layer][1].data

    @doc_inherit
    def get_layeroutput(self, layer):
        assert self.ready, 'Forward has not been called. Layer outputs are not ready.'
        if layer not in self.net.blobs:
            return None
        return self.net.blobs[layer].data

    def preprocess(self, listofimages):
        """
        Preprocess a list of images to be used with the neural network.

        Parameters
        ----------
        listofimages : List of strings or list of ndarrays, shape (Height, Width, Channels)
            The list may contain image filepaths and image ndarrays.
            For ndarrays, the shape (Height, Width, Channels) has to conform with the input size stated in the model prototxt.
            ndarrays have to be normalized to 1.

        Returns
        -------
        output : ndarray
            Preprocessed batch of images.
        """
        # transform input
        shape = self.net.blobs['data'].shape
        np_shape = [shape[i] for i in range(len(shape))]
        np_shape[0] = len(listofimages)

        data = np.zeros(np_shape)

        for i, h in enumerate(listofimages):
            if type(h) is str:
                data[i] = self.transformer.preprocess('data', caffe.io.load_image(h))
            elif type(h) is np.ndarray:
                data[i] = self.transformer.preprocess('data', h)

        return data

    @doc_inherit
    def forward(self, data):
        self.net.blobs['data'].reshape(*data.shape)
        self.net.blobs['data'].data[...] = data[...]
        out = self.net.forward()

        self.ready = True

        clean_out = []
        for k, v in out.items():
            clean_out.append(v)
        out = clean_out

        if len(out) == 1:
            return out[0]
        else:
            return out

    @doc_inherit
    def set_weights(self, layer, weights):
        if layer not in self.net.params:
            raise KeyError('Layer {} does not exist.'.format(layer))

        param_shape = tuple(self.net.params[layer][0].shape)
        if param_shape != weights.shape:
            raise ValueError('Weight dimensions ({}, {}) do not match.'.format(
                str(param_shape),
                str(weights.shape)))

        self.net.params[layer][0].data[...] = weights[...]

    @doc_inherit
    def set_bias(self, layer, bias):
        if layer not in self.net.params:
            raise KeyError('Layer {} does not exist.'.format(layer))

        param_shape = tuple(self.net.params[layer][1].shape)
        if param_shape != bias.shape:
            raise ValueError('Bias dimensions ({}, {}) do not match.'.format(
                str(param_shape),
                str(bias.shape)))

        self.net.params[layer][1].data[...] = bias[...]
