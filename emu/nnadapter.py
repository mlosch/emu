import numpy as np
import image


class NNAdapter(object):
    """
    Base class for NN library interfaces to load and read pre-trained models.

    Override the abstract methods for different neural network libraries like Caffe, Torch or Keras,
    to enable simplified communication between analysis scripts and libraries.
    """

    def forward(self, input):
        """
        Forward a preprocessed batch of images through the neural network.

        Parameters
        ----------
        input : ndarray
            Preprocessed input.

        Returns
        -------
        output : ndarray
            Output of final network layer.
        """
        raise NotImplementedError

    def set_weights(self, layer, weights):
        """
        Set the weights of a particular layer to manually designed values.
        Can be utilized to lesion or augment specific units and or parameters.

        Note that the change is permanent. To conserve the original state,
        get the parameters via get_layerparams() beforehand.

        Parameters
        ----------
        layer : string
            Layer identification.

        weights : numpy.ndarray.
            Weight array with same dimensionality as the targeted existing layer weight.
            See get_layerparams() to determine the dimensions of a specific layer.
        """
        raise NotImplementedError

    def set_bias(self, layer, bias):
        """
        Set the bias of a particular layer to manually designed values.
        Can be utilized to lesion or augment specific units and or parameters.

        Note that the change is permanent. To conserve the original state,
        get the parameters via get_layerparams() beforehand.

        Parameters
        ----------
        layer : string
            Layer identification.

        bias : numpy.ndarray.
            Bias array with same dimensionality as the targeted existing layer weight.
            See get_layerparams() to determine the dimensions of a specific layer.
        """
        raise NotImplementedError

    def get_layerparams(self, layer):
        """
        Get the parameters of a specific layer.

        Parameters
        ----------
        layer : string
            Layer identification.

        Returns
        -------
        (weights, bias) : Tuple of ndarrays
        """
        raise NotImplementedError

    def get_layeroutput(self, layer):
        """
        Get the output of a specific layer.
        forward(...) has to be called in advance.

        Parameters
        ----------
        layer : string, Layer identification
            Specifying the location of the layer within the model.
            To see all identifiers, call `get_layers`.

        Returns
        -------
        output : ndarray
            Numpy tensor of output values.

        Raises
        -------
        ValueError : If Layer is not defined in model
        """
        raise NotImplementedError

    def get_layers(self):
        """
        Get layer ids/names and their corresponding layer type of
        all layers within the loaded network.

        Returns
        -------
        Layer info : OrderedDict
            Dictionary of layer ids/names and corresponding types.
        """
        raise NotImplementedError

    def model_description(self):
        """
        Get a formatted model description as string containing layer names of the model architecture.

        Returns
        -------
        Description : string
        """
        raise NotImplementedError

    @staticmethod
    def preprocess(listofimages, inputsize, mean, std, dimorder='chw', channelorder='rgb', scale=1.0):
        """
        Preprocess a list of images to be used with the neural network.
        Automatically loads images from disk if a string is given.

        Parameters
        ----------
        listofimages : List of strings or list of unprocessed ndarrays, shape (Height, Width, Channels)
            The list may contain image filepaths and image ndarrays.
            ndarrays have to be in range between 0 and 1.
        inputsize : tuple or list
            Target input data dimensionality of same format as dimorder
        mean : ndarray
            Image mean definition of format (channels, height, width) or (channels, ).
            Set to None, to ignore.
        std : ndarray
            Standard deviation definition of format (channels, height, width) or (channels, )
            Set to None, to ignore
        dimorder : string
            Order of dimensions. Default chw (channel, height, width)
        channelorder : string
            Order of color channels. Default: rgb (red, green, blue)
        scale : float
            Scaling of color values.

        Returns
        -------
        output : ndarray
            Preprocessed batch of images ready for the forward pass.
        """

        # Convert dimorder to tuple
        dimmap = {
            'h': 0,
            'w': 1,
            'c': 2,
        }
        imsize = (inputsize[dimorder.find('h')], inputsize[dimorder.find('w')])
        dimorder = [dimmap[c] for c in dimorder]

        # Load data in first step from list
        data = np.zeros((len(listofimages), inputsize[0], inputsize[1], inputsize[2]), dtype=np.float32)

        for i, h in enumerate(listofimages):
            if type(h) is str:
                im = image.read(h)
            elif type(h) is np.ndarray:
                im = h.copy()

            if im.shape[:2] != imsize:
                im = image.resize(im, imsize)

            if scale != 1.0:
                im *= scale

            if mean is not None and mean.ndim == 1:
                im -= mean
            if std is not None and std.ndim == 1:
                im /= std

            if channelorder == 'bgr':
                im = im[:, :, ::-1]

            im = im.transpose(*dimorder)  # resulting order is: channels x height x width
            if mean is not None and mean.ndim == 3:
                im -= mean
            if std is not None and std.ndim == 3:
                im /= std

            data[i] = im

        return data