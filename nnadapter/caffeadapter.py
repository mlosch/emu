from nnadapter import NNAdapter
from collections import OrderedDict
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np


class CaffeAdapter(NNAdapter):
    """
    Overrides the NNAdapter to load and read Caffe models.
    An installation of Caffe and pycaffe is required.
    """
    def __init__(self, prototxt, caffemodel, mean):
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

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

    def model_description(self):
        for k, v in self.net.blobs.items():
            print(k, v)

    def get_layers(self):
        """
        Get layer ids/names and their corresponding layer type of
        all layers within the loaded network.

        Returns
        -------
        Layer info : OrderedDict
            Dictionary of layer ids/names and corresponding types.
        """
        return self.layer_types

    def get_layerparams(self, layer):
        """
        Get the parameters of a specific layer.

        Parameters
        ----------
        layer : String
            Layer identification.

        Returns
        -------
        (weights, bias) : Tuple of ndarrays
        """
        if layer not in self.net.params:
            return None
        return self.net.params[layer][0].data, self.net.params[layer][1].data

    def get_layeroutput(self, layer):
        assert self.ready, 'Forward has not been called. Layer outputs are not ready.'
        if layer not in self.net.blobs:
            return None
        return self.net.blobs[layer].data

    def forward(self, input):
        # transform input
        shape = self.net.blobs['data'].shape
        np_shape = [shape[i] for i in range(len(shape))]
        np_shape[0] = len(input)

        data = np.zeros(np_shape)

        for i, h in enumerate(input):
            if type(h) == str:
                data[i] = self.transformer.preprocess('data', caffe.io.load_image(h))
            elif type(h) == np.ndarray:
                data[i] = self.transformer.preprocess('data', h)

        self.net.blobs['data'].reshape(*data.shape)
        self.net.blobs['data'].data[...] = data[...]
        out = self.net.forward()

        self.ready = True

        return out[out.keys()[0]]
