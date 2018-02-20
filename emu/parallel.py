import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from nnadapter import NNAdapter


def _preprocessor(args):
    i, images, inputsize, mean, std = args
    return i, NNAdapter.preprocess(images, inputsize, mean, std)


def _imap_generator(net, imagelist, batch_size):
    for k, i in enumerate(range(0, len(imagelist), batch_size)):
        yield k, imagelist[i:i+batch_size], net.inputsize, net.mean, net.std


def forward(net, imagelist, keep_outputs, batch_size=128, processes=4, progress=False):
    """
    Parallel data loading and feed forwarding of a list of images through the neural network.

    Parameters
    ----------
    net : NNAdapter
        emu based model
    imagelist : List of strings or list of unprocessed ndarrays, shape (Height, Width, Channels)
        The list may contain image filepaths and image ndarrays.
        ndarrays have to be in range between 0 and 1.
    keep_outputs : List of strings
        Defines which layer outputs are returned.
    batch_size : int
        Default value: 128
    processes : int
        Defines the number of processes that load image data concurrently.
        Default value: 4
    progress : bool
        Whether to show a tqdm progress bar.
        Default value: False

    Returns
    -------
    output : dict
        Dictionary containing the output of each requested network layer.
    """
    if processes < 1:
        raise ValueError('Number of processes cannot be less than 1.')
    if batch_size < 1:
        raise ValueError('Batchsize cannot be less than 1.')

    out_target = {}
    for layer in keep_outputs:
        out_target[layer] = {}

    pbar = tqdm if progress else lambda x, **kwargs: x

    p = mp.Pool(processes=min(processes, int(np.ceil(len(imagelist)/float(batch_size)))))

    args = _imap_generator(net, imagelist, batch_size)
    inds = range(0, len(imagelist), batch_size)

    try:
        for i, batch in pbar(p.imap_unordered(_preprocessor, args), total=len(inds), desc='Forward', unit='Batch'):
            net.forward(batch)

            for layer in keep_outputs:
                out_target[layer][i] = net.get_layeroutput(layer)
    except KeyboardInterrupt as e:
        p.terminate()
        p.join()
        raise e

    p.close()
    p.join()

    for layer in keep_outputs:
        out_list = [out_target[layer][i] for i in range(len(out_target[layer]))]
        out_target[layer] = np.concatenate(out_list, axis=0)

    return out_target

