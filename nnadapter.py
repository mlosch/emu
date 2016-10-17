class NNAdapter(object):
    """
    Base class for NN library interfaces to load and read pre-trained models.

    By overriding the abstract methods for different neural network libraries like Caffe, Torch or Keras,
    we can enable simplified communication between analysis scripts and libraries.
    """
    def forward(self, input):
        """
        Forward a batch of images through the neural network.

        Parameters
        ----------
        input : List of strings or list of ndarrays.
            The list may contain image filepaths and image ndarrays.

        Returns
        -------
        output : ndarray
            Output of final network layer.
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
