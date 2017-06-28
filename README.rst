.. role:: python(code)
   :language: python

emu
===
Generic, lightweight python interface to unify loading and evaluating neural networks trained in machine learning libraries such as Keras, PyTorch, Torch7 and Caffe.

What is emu?
------------

emu is a lightweight interface to deep learning libraries to simplify the creation of analysis toolchains. The focus is on processing images with pretrained image recognition models, manipulating parameters (e.g. to lesion a set of units) as well as reading parameters and output values of arbitrary layers. Hence, no training is supported with emus interface.
Currently supported machine learning libraries (backends) are:

- `Keras <https://keras.io/>`_
- `pytorch <http://pytorch.org>`_
- `Torch7 <http://torch.ch>`_
- `Caffe <http://caffe.berkeleyvision.org>`_

Core functionality
------------------
emus design pattern is the `adapter pattern <https://en.wikipedia.org/wiki/Adapter_pattern>`_, meaning it converts the interface of the backend libraries to a unified interface :python:`NNAdapter` with the following core functionality:

- :python:`forward(4d_numpy_tensor)`
    Process a batch of images.
- :python:`preprocess(list_of_images)`
    Preprocess a list of images for use with the neural network.
- :python:`get_layers()`
    Get an ordered dictionary assigning a layer type to each layer name. The order is equal to the sequence of layers within that model.
- :python:`get_layeroutput(layer_name)`
    After `forward` has been called, get the output of layer with name `layer_name`.
- :python:`get_layerparams(layer_name)`
    Get the weights and bias parameter of layer with name `layer_name`.
- :python:`set_weights(layer_name, weight_tensor)`
    Assign new weights to the given layer
- :python:`set_bias(layer_name, bias_tensor)`
    Assign a new bias to the given layer

To use emu with a specific backend, you have to instantiate the respective backend class, as the :python:`NNAdapter` does itself not contain any functionality besides a generic preprocessing function.
Overview over implemented classes inheriting :python:`NNAdapter`:

- Keras: :python:`KerasAdapter`
- pytorch/Torch7: :python:`TorchAdapter`
- Caffe: :python:`CaffeAdapter`

The instantiation of these classes slightly differ as they depend on the quirks of the backends.
Please see the documentation of these classes or have a look at the example notebooks at the bottom of this readme.

Minimal example
---------------
Forward two images through a Keras model and read the output of the first convolutional layer.

.. code:: python

    import emu
    from emu.keras import KerasAdapter

    # initialize model from Keras stock model zoo
    mean = emu.keras.imagenet_mean
    nn = KerasAdapter('ResNet50', 'imagenet', mean, std=None, inputsize=(224,224,3), keep_outputs=['conv1'], use_gpu=True)

    # define two example images
    #  Note that it is sufficient to pass file paths.
    #  The preprocess() function automatically loads the images with RGB color channels.
    images = ['data/images/img1.jpg',
              'data/images/img5.png']

    # preprocess and forward images
    batch = nn.preprocess(images)
    predictions = nn.forward(batch)

    # read output of conv1
    conv1_output = nn.get_layeroutput('conv1')

    # get weights of conv1
    weight, bias = nn.get_layerparams('conv1')

Prerequisites
-------------

- General
    - numpy (>= 1.11.1)
    - scikit-image (>= 0.12.3)
- Using Keras as backend
    - `Keras <https://keras.io/#installation>`_
    - and one of:
        - `TensorFlow <https://www.tensorflow.org/install/>`_
        - `Theano <http://deeplearning.net/software/theano/install.html>`_
        - `CNTK <https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine>`_
- Using pytorch/torch7 as backend
    - `PyTorch <https://github.com/pytorch/pytorch#installation>`_
- Using Caffe as backend
    - `Caffe <http://caffe.berkeleyvision.org/installation.html>`_
    - `pyCaffe <http://caffe.berkeleyvision.org/installation.html#python-andor-matlab-caffe-optional>`_

Installation
------------

.. code:: shell

    python setup.py install
    
How-To
------

- Find pretrained models:
    - **Keras:**
        - `Model Zoo <https://keras.io/applications/>`_
          After installation, use pretrained models via passing an available architecture name to the `KerasAdapter`, 
          e.g.: :python:`KerasAdapter(model_cfg='ResNet50', model_weights='imagenet')`. See `Available models <https://keras.io/applications/#available-models>`_
    - **Caffe:** 
        - `Model Zoo <https://github.com/BVLC/caffe/wiki/Model-Zoo>`_
        - `ResNets <https://github.com/KaimingHe/deep-residual-networks#models>`_
    - **PyTorch:**
        - `Model Zoo <https://github.com/pytorch/vision#installation>`_
          After installation, use pretrained models via passing an available architecture name to the `TorchAdapter`, 
          e.g.: :python:`TorchAdapter(model_fp='resnet18')`. See `Available models <https://github.com/pytorch/vision#models>`_
    - **Torch7:** (Warning, support is rudimentary) 
        - `Model Zoo <https://github.com/torch/torch7/wiki/ModelZoo>`_
        - `ResNets <https://github.com/facebook/fb.resnet.torch/tree/master/pretrained>`_
        
Example notebooks
-----------------
- `Using emu to estimate mean and standard deviation <examples/summary_statistics.ipynb>`_ of pretrained caffe or torch models.
- `Lesioning/Altering parameters <examples/evaluate_and_lesion.ipynb>`_ of models


Why the name emu?
-----------------
This package is named after the bird, which as the functionality in this package cannot run backwards.