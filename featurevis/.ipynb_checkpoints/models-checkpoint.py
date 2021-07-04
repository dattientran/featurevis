import torch

from featurevis.utils import varargin


class Ensemble():
    """ Average the response across a set of models.

    Arguments:
        models (list): A list of pytorch models.
        readout_key (str): String identifying the scan whose neurons will be outputted by
            the model
        eye_pos (torch.Tensor): A [1 x 2] tensor with the position of the pupil(x, y).
            This shifts the FOV of all cells. Default (None) is position at center of
            screen (i.e., it disables the shifter).
        behavior (torch.Tensor): A [1 x 3] tensor with the behavior parameters
            (pupil_dilation, dpupil_dilation/dt, treadmill). Default is to return results
            without modulation.
        neuron_idx (int or slice or list): Neuron(s) to return. Default returns all cells.
        y_shift, x_shift (float or torch tensor): Overwrite the learnt (per-cell readout)
            shift with these values for all cells. Values are clipped to [-1, 1] (see
            torch.nn.functional.grid_sample). Default uses the learnt readout shift.
        average_batch (boolean): If True, responses are averaged across all images in the
            batch (output is a num_neurons tensor). Otherwise, output is a num_images x
            num_neurons tensor.
        device (torch.Device or str): Where to load the models.

    Note:
        We copy the models to avoid overwriting the gradients (and grid if x_shift or
        y_shift is set) of the original models. You can access our copy of the models as
        my_ensemble.models.
    """
    def __init__(self, models, readout_key, eye_pos=None, behavior=None,
                 neuron_idx=slice(None), y_shift=None, x_shift=None,
                 average_batch=True, device='cuda'):
        import copy
        self.models = []
        for m in models:
            m_copy = type(m)(core=m.core, readout=m.readout, modulator=m.modulator, nonlinearity=m.nonlinearity, shifter=m.shifter)
            m_copy.load_state_dict(m.state_dict())
            self.models.append(m_copy)
#         self.models = [copy.deepcopy(m) for m in models]
        self.readout_key = readout_key
        self.eye_pos = None if eye_pos is None else eye_pos.to(device)
        self.behavior = None if behavior is None else behavior.to(device)
        self.neuron_idx = neuron_idx
        self.y_shift = y_shift
        self.x_shift = x_shift
        self.average_batch = average_batch
        self.device = device

        for m in self.models:
            # If needed, change readout shifts to the desired ones
            with torch.no_grad():
                if self.y_shift is not None:
                    m.readout[readout_key].grid[..., 1] = self.y_shift
                if self.x_shift is not None:
                    m.readout[readout_key].grid[..., 0] = self.x_shift

            m.to(device)
            m.eval()

    def __call__(self, x):
        resps = [m(x, self.readout_key, eye_pos=self.eye_pos, behavior=self.behavior)[:,
                 self.neuron_idx] for m in self.models]
        resps = torch.stack(resps)  # num_models x batch_size x num_neurons
        resp = resps.mean(0).mean(0) if self.average_batch else resps.mean(0)

        return resp


class VGG19Core():
    """ A pretrained VGG-19. Output will be intermediate feature representation
    (N x C x H x W) at the desired layer.

    Arguments:
        layer (int): Index (0-based) of the layer that will be optimized.
        use_batchnorm (boolean): Whether to download the version with batchnorm.
        device (torch.Device or str): Where to place the model.
    """
    def __init__(self, layer, use_batchnorm=True, device='cuda'):
        from torchvision import models

        vgg19 = (models.vgg19_bn(pretrained=True) if use_batchnorm else
                 models.vgg19(pretrained=True))
        if layer < len(vgg19.features):
            self.model = vgg19.features[:layer + 1]
        else:
            raise ValueError('layer out of range (max is', len(vgg19.features))
        self.model.to(device)
        self.model.eval()

    @varargin
    def __call__(self, x):
        return self.model(x)


class VGG19():
    """ A pretrained VGG-19. Output will be the average of one channel across spatial
    dimensions.

    Arguments:
        layer (int): Index (0-based) of the layer that will be optimized.
        channel (int)_: Index (0-based) of the channel that will be optimized.
        use_batchnorm (boolean): Whether to download the version with batchnorm.
        device (torch.Device or str): Where to place the model.
    """
    def __init__(self, layer, channel, use_batchnorm=True, device='cuda'):
        self.model = VGG19Core(layer, use_batchnorm, device)
        self.channel = channel

    def __call__(self, x):
        resp = self.model(x)[:, self.channel, :, :].mean()
        return resp
