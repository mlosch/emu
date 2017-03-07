class NNAdapter(object):
    """
    Base class for NN library interfaces to load and read pre-trained models.

    Override the abstract methods for different neural network libraries like Caffe, Torch or Keras,
    to enable simplified communication between analysis scripts and libraries.
    """
    def preprocess(self, listofimages):
        """
        Preprocess a list of images to prepare it for use in the neural network.
        Automatically loads images from disk if a string is given.

        Parameters
        ----------
        input : List of strings or list of ndarrays.
            The list may contain image filepaths and image ndarrays.

         Returns
        -------
        output : ndarray
            Preprocessed batch of images ready for the forward pass.
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
        layer : String
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
        layer : String
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
        layer : String
            Layer identification.

        Returns
        -------
        (weights, bias) : Tuple of ndarrays
        """
        raise NotImplementedError

    def get_layeroutput(self, layer):
        """
        Get the output of a specific layer.

        Parameters
        ----------
        layer : String, Layer identification

        Returns
        -------
        output : ndarray
            Numpy tensor of output values.
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
        Prints a model description to stdout displaying layer names of the model architecture.
        """
        raise NotImplementedError
