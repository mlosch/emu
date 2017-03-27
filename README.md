# nnadapter
Generic python interface to unify access to neural networks trained in machine learning libraries such as Caffe, PyTorch and Torch7.

## What is nnadapter?

NNAdapter is a lightweight interface to deep learning libraries to simplify the creation of analysis toolchains. The focus is on processing images with pretrained image recognition models, manipulating parameters (e.g. to lesion a set of units) as well as reading parameters and output values of arbitrary layers. Hence, no training is supported with the nnadapter-interface.
Currently supported machine learning libraries (backends) are:

- [Caffe](http://caffe.berkeleyvision.org)
- [pytorch](http://pytorch.org) & [Torch7](http://torch.ch)

## Core functionality
The base class defines the following functionality:

- `forward(4d_numpy_tensor)`
    - Process a batch of images.
- `preprocess(list_of_images)`
    - Preprocess a list of images for use with the neural network.
- `get_layers()`
    - Get an ordered dictionary assigning a layer type to each layer name. The order is equal to the sequence of layers within that model.
- `get_layeroutput(layer_name)`
    - After `forward` has been called, get the output of layer with name `layer_name`.
- `get_layerparams(layer_name)`
    - Get the weights and bias parameter of layer with name `layer_name`.
- `set_weights(layer_name, weight_tensor)`
    - Assign new weights to the given layer
- `set_bias(layer_name, bias_tensor)`
    - Assign a new bias to the given layer
    

## Prerequisites

- General
    - numpy (>= 1.11.1)
    - scikit-image (>= 0.12.3)
- Using Caffe as backend
    - [caffe](http://caffe.berkeleyvision.org/installation.html)
    - [pycaffe](http://caffe.berkeleyvision.org/installation.html#python-andor-matlab-caffe-optional)
- Using torch/pytorch as backend
    - [pytorch](https://github.com/pytorch/pytorch#installation)

## Installation

`python setup.py install`
    
## How-To

- Find pretrained models:
    - **Caffe:** 
        - [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
        - [ResNets](https://github.com/KaimingHe/deep-residual-networks#models)
    - **PyTorch:**
        - [Model Zoo](https://github.com/pytorch/vision#installation)
          After installation, use pretrained models via passing an available architecture name to the `TorchAdapter`, 
          e.g.: `TorchAdapter(model_fp='resnet18')`. See [Available models](https://github.com/pytorch/vision#models)
    - **Torch7:** (Warning, support is rudimentary) 
        - [Model Zoo](https://github.com/torch/torch7/wiki/ModelZoo)
        - [ResNets](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)
        
## Example notebooks  
- [Loading and using NNAdapter](examples/summary_statistics.ipynb) on how to use NNAdapter to estimate mean and standard deviation of pretrained caffe or torch models.
- [Lesioning/Altering parameters](examples/evaluate_and_lesion.ipynb) of models