import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy import signal
from scipy.stats import multivariate_normal

from featurevis.utils import varargin



################################## REGULARIZERS ##########################################
class DoNothing():
    @varargin
    def __call__(self, x):
        return x

class ReverseSign():
    @varargin
    def __call__(self, x):
        return - x

class Feature_Vector_Ensemble():
    def __init__(self, models, readout_key, eye_pos=None, behavior=None, neuron_idx=slice(None), average_batch=True, device='cuda'):
        import copy
        
        self.models = [copy.deepcopy(m) for m in models]
        self.readout_key = readout_key
        self.eye_pos = None if eye_pos is None else eye_pos.to(device)
        self.behavior = None if behavior is None else behavior.to(device)
        self.neuron_idx = neuron_idx
        self.average_batch = average_batch
        self.device = device

    def __call__(self, x, iteration=None):
        vecs = []
        
        for m in self.models:
            m.to(self.device)
            m.eval()
            
            def feature_vector_forward(x, neuron_idx=self.neuron_idx, self=m.readout[self.readout_key], shift=None):
                if self.positive:
                    positive(self.features)
                self.grid.data = torch.clamp(self.grid.data, -1, 1)
                N, c, w, h = x.size()
                m = self.gauss_pyramid.scale_n + 1
                feat = self.features.view(1, m * c, self.outdims)

                if shift is None:
                    grid = self.grid.expand(N, self.outdims, 1, 2)
                else:
                    grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]
                pools = [F.grid_sample(xx, grid, align_corners=True)[:, :, neuron_idx, :] for xx in self.gauss_pyramid(x)]
                y = torch.cat(pools, dim=1).squeeze()
                return y     
            
            m.readout[self.readout_key].forward = feature_vector_forward  # monkey patching the original readout forward
            # m.modulator = None # avoids using the modulator on the output of readout (this is part of the forward of the model in base.py)
            # m.nonlinearity = DoNothing() # to avoid applying the nonlinearity in the forward of the model
            vecs.append(m(x, self.readout_key, eye_pos=self.eye_pos, behavior=self.behavior))
            
        vecs = torch.stack(vecs) # num_models * batch_size * feature_vec_length
        vecs = vecs.mean(0).mean(0) if self.average_batch else vecs.mean(0)

        return vecs

class SingleGridResps():
    def __init__(self, models, readout_key, eye_pos=None, behavior=None, fixed_grid=None, neuron_idx=slice(None), average_batch=True, all_neurons=True, device='cuda'):
        import copy
        
        self.models = [copy.deepcopy(m) for m in models]
        self.readout_key = readout_key
        self.eye_pos = None if eye_pos is None else eye_pos.to(device)
        self.behavior = None if behavior is None else behavior.to(device)
        self.fixed_grid = fixed_grid
        self.neuron_idx = neuron_idx
        self.average_batch = average_batch
        self.all_neurons = all_neurons
        self.device = device

    def __call__(self, x, iteration=None):
        resps = []
        
        for m in self.models:
            m.to(self.device)
            m.eval()
            
            def single_grid_forward(x, fixed_grid=self.fixed_grid, neuron_idx=self.neuron_idx, self=m.readout[self.readout_key], shift=None):
                if self.positive:
                    positive(self.features)
                self.grid.data = torch.clamp(self.grid.data, -1, 1)
                N, c, w, h = x.size()
                m = self.gauss_pyramid.scale_n + 1
                feat = self.features.view(1, m * c, self.outdims)

                if shift is None:
                    grid = self.grid.expand(N, self.outdims, 1, 2)
                else:
                    grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]

                # Alls neuron read from the grid of the target neuron
                if fixed_grid is not None: 
                    grid = torch.as_tensor(fixed_grid, dtype=torch.float32)[None, None, None].expand(N, self.outdims, 1, 2).to('cuda')
                elif neuron_idx is not None:
                    grid = grid[:, neuron_idx, :, :][:, None, :, :].expand(N, self.outdims, 1, 2)
                else: 
                    raise Exception("fixed_grid and neuron_idx cannot be None at the same time!")

                pools = [F.grid_sample(xx, grid, align_corners=True)[:, :, neuron_idx:neuron_idx+1, :] for xx in self.gauss_pyramid(x)]
                y = torch.cat(pools, dim=1).squeeze(-1)
                y = (y * feat).sum(1).view(N, self.outdims)

                if self.bias is not None:
                    y = y + self.bias

                return y     
            
            m.readout[self.readout_key].forward = single_grid_forward  # monkey patching the original readout forward
            if self.all_neurons:
                resps.append(m(x, self.readout_key, eye_pos=self.eye_pos, behavior=self.behavior))
            else:
                resps.append(m(x, self.readout_key, eye_pos=self.eye_pos, behavior=self.behavior)[:, self.neuron_idx])

        resps = torch.stack(resps)  # num_models x batch_size x num_neurons
        resp = resps.mean(0).mean(0) if self.average_batch else resps.mean(0)
        return resp

class TotalVariation():
    """ Total variation regularization.

    Arguments:
        weight (float): Weight of the regularization.
        isotropic (bool): Whether to use the isotropic or anisotropic definition of Total
            Variation. Default is anisotropic (l1-norm of the gradient).
    """
    def __init__(self, weight=1, isotropic=False):
        self.weight = weight
        self.isotropic = isotropic

    @varargin
    def __call__(self, x):
        # Using the definitions from Wikipedia.
        diffs_y = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        diffs_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        if self.isotropic:
            tv = torch.sqrt(diffs_y[:, :, :, :-1] ** 2 +
                            diffs_x[:, :, :-1, :] ** 2).reshape(len(x), -1).sum(-1)  # per image
        else:
            tv = diffs_y.reshape(len(x), -1).sum(-1) + diffs_x.reshape(len(x), -1).sum(-1)  # per image
        loss = self.weight * torch.mean(tv)

        return loss


class LpNorm():
    """Computes the lp-norm of an input.

    Arguments:
        weight (float): Weight of the regularization
        p (int): Degree for the l-p norm.
    """
    def __init__(self, weight=1, p=6):
        self.weight = weight
        self.p = p

    @varargin
    def __call__(self, x):
        lpnorm = (torch.abs(x) ** self.p).reshape(len(x), -1).sum(-1) ** (1 / self.p)
        loss = self.weight * torch.mean(lpnorm)
        return loss


class Similarity():
    """ Compute similarity metrics across all examples in one batch.

    Arguments:
        weight (float): Weight of the regularization.
        metric (str): What metric to use when computing pairwise similarities. One of:
            correlation: Masked correlation.
            cosine: Cosine similarity of the masked input.
            neg_euclidean: Negative of euclidean distance between the masked input.
        combine_op (function): Function used to agglomerate pairwise similarities.
        mask (torch.tensor or None): Mask to use when calculating similarities. Expected
            to be in [0, 1] range and be broadcastable with input.
    """
    def __init__(self, weight=1, metric='correlation', combine_op=torch.max, mask=None):
        self.weight = weight
        self.metric = metric
        self.combine_op = combine_op
        self.mask = mask

    @varargin
    def __call__(self, x):
        if len(x) < 2:
            warnings.warn('Only one image in the batch. Similarity regularization will'
                          'return 0')
            return 0

        # Mask x
        masked_x = x if self.mask is None else x * self.mask
        flat_x = masked_x.view(len(x), -1)

        # Compute similarity matrix
        if self.metric == 'correlation':
            if self.mask is None:
                residuals = flat_x - flat_x.mean(-1, keepdim=True)
                numer = torch.mm(residuals, residuals.t())
                ssr = (residuals ** 2).sum(-1)
            else:
                mask_sum = self.mask.sum() * (flat_x.shape[-1] / len(self.mask.reshape(-1)))
                mean = flat_x.sum(-1) / mask_sum
                residuals = x - mean.view(len(x), *[1, ] * (x.dim() - 1))  # N x 1 x 1 x 1
                numer = (residuals[None, :] * residuals[:, None] * self.mask).view(
                    len(x), len(x), -1).sum(-1)
                ssr = ((residuals ** 2) * self.mask).view(len(x), -1).sum(-1)
            sim_matrix = numer / (torch.sqrt(torch.ger(ssr, ssr)) + 1e-9)
        elif self.metric == 'cosine':
            norms = torch.norm(flat_x, dim=-1)
            sim_matrix = torch.mm(flat_x, flat_x.t()) / (torch.ger(norms, norms) + 1e-9)
        elif self.metric == 'neg_euclidean':
            sim_matrix = -torch.norm(flat_x[None, :, :] - flat_x[:, None, :], dim=-1)
        elif self.metric == 'mse':
            mask_sum = self.mask.sum() * (flat_x.shape[-1] / len(self.mask.reshape(-1)))
            sim_matrix = - (torch.norm(flat_x[None, :, :] - flat_x[:, None, :], dim=-1) **2 / mask_sum)
        else:
            raise ValueError('Invalid metric name:{}'.format(self.metric))

        # Compute overall similarity
        triu_idx = torch.triu(torch.ones(len(x), len(x)), diagonal=1) == 1
        similarity = self.combine_op(sim_matrix[triu_idx])

        loss = self.weight * similarity

        return loss


# class PixelCNN():
#     def __init__(self, weight=1):
#         self.weight = weight
#
#         self.pixel_cnn = ... # load the model
#
#     @varargin
#     def __call__(self, x):
#         # Modify x to make it a valid input to pixel cnn (add channels)
#         prior = self.pixel_cnn(x)
#         loss = self.weight * prior


################################ TRANSFORMS ##############################################
class Jitter():
    """ Jitter the image at random by some certain amount.

    Arguments:
        max_jitter(tuple of ints): Maximum amount of jitter in y, x.
    """
    def __init__(self, max_jitter):
        self.max_jitter = max_jitter if isinstance(max_jitter, tuple) else (max_jitter,
                                                                            max_jitter)

    @varargin
    def __call__(self, x):
        # Sample how much to jitter
        jitter_y = torch.randint(-self.max_jitter[0], self.max_jitter[0] + 1, (1,),
                                 dtype=torch.int32).item()
        jitter_x = torch.randint(-self.max_jitter[1], self.max_jitter[1] + 1, (1,),
                                 dtype=torch.int32).item()

        # Pad and crop the rest
        pad_y = (jitter_y, 0) if jitter_y >= 0 else (0, -jitter_y)
        pad_x = (jitter_x, 0) if jitter_x >= 0 else (0, -jitter_x)
        padded_x = F.pad(x, pad=(*pad_x, *pad_y), mode='reflect')

        # Crop
        h, w = x.shape[-2:]
        jittered_x = padded_x[..., slice(0, h) if jitter_y > 0 else slice(-jitter_y, None),
                              slice(0,w) if jitter_x > 0 else slice(-jitter_x, None)]

        return jittered_x


class RandomCrop():
    """ Take a random crop of the input image.

    Arguments:
        height (int): Height of the crop.
        width (int): Width of the crop
    """
    def __init__(self, height, width, n_crops=1):
        self.height = height
        self.width = width
        self.n_crops = n_crops

    @varargin
    def __call__(self, x):
        crop_y = torch.randint(0, max(0, x.shape[-2] - self.height) + 1, (self.n_crops,),
                               dtype=torch.int32)
        crop_x = torch.randint(0, max(0, x.shape[-1] - self.width) + 1, (self.n_crops,),
                               dtype=torch.int32)
        crops = []
        for cy, cx in zip(crop_y, crop_x):
            cropped_x = x[..., cy: cy + self.height, cx: cx + self.width]
            crops.append(cropped_x)
            
        return torch.vstack(crops)


class BatchedCrops():
    """ Create a batch of crops of the original image.

    Arguments:
        height (int): Height of the crop
        width (int): Width of the crop
        step_size (int or tuple): Number of pixels in y, x to step for each crop.
        sigma (float or tuple): Sigma in y, x for the gaussian mask applied to each batch.
            None to avoid masking

    Note:
        Increasing the stride of every convolution to stride * step_size produces the same
        effect in a much more memory efficient way but it will be architecture dependent
        and may not play nice with the rest of transforms.
    """
    def __init__(self, height, width, step_size, sigma=None):
        self.height = height
        self.width = width
        self.step_size = step_size if isinstance(step_size, tuple) else (step_size,) * 2
        self.sigma = sigma if sigma is None or isinstance(sigma, tuple) else (sigma,) * 2

        # If needed, create gaussian mask
        if sigma is not None:
            y_gaussian = signal.gaussian(height, std=self.sigma[0])
            x_gaussian = signal.gaussian(width, std=self.sigma[1])
            self.mask = y_gaussian[:, None] * x_gaussian

    @varargin
    def __call__(self, x):
        if len(x) > 1:
            raise ValueError('x can only have one example.')
        if x.shape[-2] < self.height or x.shape[-1] < self.width:
            raise ValueError('x should be larger than the expected crop')

        # Take crops
        crops = []
        for i in range(0, x.shape[-2] - self.height + 1, self.step_size[0]):
            for j in range(0, x.shape[-1] - self.width + 1, self.step_size[1]):
                crops.append(x[..., i: i + self.height, j: j + self.width])
        crops = torch.cat(crops, dim=0)

        # Multiply by a gaussian mask if needed
        if self.sigma is not None:
            mask = torch.as_tensor(self.mask, device=crops.device, dtype=crops.dtype)
            crops = crops * mask

        return crops


class ChangeRange():
    """ This changes the range of x as follows:
        new_x = sigmoid(x) * (desired_max - desired_min) + desired_min

    Arguments:
        x_min (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
        x_max (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
    """
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x):
        new_x = torch.sigmoid(x) * (self.x_max - self.x_min) + self.x_min
        return new_x


class Resize():
    """ Resize images.

    Arguments:
        scale_factor (float): Factor to rescale the images:
            new_h, new_w = round(scale_factor * (old_h, old_w)).
        resize_method (str): 'nearest' or 'bilinear' interpolation.

    Note:
        This changes the dimensions of the image.
    """
    def __init__(self, scale_factor, resize_method='bilinear'):
        self.scale_factor = scale_factor
        self.resample_method = resize_method

    @varargin
    def __call__(self, x):
        new_height = int(round(x.shape[-2] * self.scale_factor))
        new_width = int(round(x.shape[-1] * self.scale_factor))
        return F.upsample(x, (new_height, new_width), mode=self.resize_method)


class GrayscaleToRGB():
    """ Transforms a single channel image into three channels (by copying the channel)."""
    @varargin
    def __call__(self, x):
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError('Image is not grayscale!')

        return x.expand(-1, 3, -1, -1)


class Identity():
    """ Transform that returns the input as is."""
    @varargin
    def __call__(self, x):
        return x

class ThresholdLikelihood():
    """ Threshold the latent space of a normal distribution based on radius or likelihood.
        Arguments:
        size (int): Length of latent vector
        radius (float or tensor): Desired radius.
        likelihood (float or tensor): Desired likelihood
        fixed_radius (boolean): if True, always scale to the fixed radius regardless of original distance;
        otherwise, only scale to the desired radius if original distance is larger than desired
    """
    def __init__(self, size, radius=None,likelihood=None,fixed_radius=True):
        self.size = size
        self.fixed_radius = fixed_radius
        m = multivariate_normal(np.zeros(self.size),np.diag(np.ones((self.size,self.size))))
        if radius is not None and likelihood is not None:
            raise Exception("Only radius or likelihood can be set")
        if radius is not None:
            self.radius = radius
            temp = np.zeros(self.size)
            temp[0] = radius
            self.likelihood = m.pdf(temp)
        else:
            self.likelihood = likelihood
            def cal_radius(p,dim):
                m = multivariate_normal(0)
                p /= m.pdf(0)**(dim-1)
                x = np.sqrt(-2*np.log(p*np.sqrt(2*np.pi)))
                return x
            self.radius = cal_radius(self.likelihood,self.size)
            
    @varargin
    def __call__(self, z):
        # Scale linearly depending on the distance from origin
        dist = (z**2).sum(1).sqrt().unsqueeze(1).repeat(1,self.size)
        if self.fixed_radius:  
            scaled_z = z * (self.radius/(dist+1e-9))
        else:            
            desired_r = (dist > self.radius) * self.radius + (dist <= self.radius) * dist
            scaled_z = z * (desired_r/(dist+1e-9))
        return scaled_z

############################## GRADIENT OPERATIONS #######################################
class ChangeNorm():
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """
    def __init__(self, norm):
        self.norm = norm

    @varargin
    def __call__(self, x):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1,] * (x.dim() - 1))
        return renorm


class ClipRange():
    """Clip the value of x to some specified range.

    Arguments:
        x_min (float): Lower valid value.
        x_max (float): Higher valid value.
    """
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x):
        return torch.clamp(x, self.x_min, self.x_max)


class FourierSmoothing():
    """ Smooth the input in the frequency domain.

    Image is transformed to fourier domain, power densities at i, j are multiplied by
    (1 - ||f||)**freq_exp where ||f|| = sqrt(f_i**2 + f_j**2) and the image is brought
    back to the spatial domain:
        new_x = ifft((1 - freqs) ** freq_exp * fft(x))

    Arguments:
        freq_exp (float): Exponent for the frequency mask. Higher numbers produce more
            smoothing.

    Note:
        Consider just using Gaussian blurring. Faster and easier to explain.
    """
    def __init__(self, freq_exp):
        self.freq_exp = freq_exp

    @varargin
    def __call__(self, x):
        # Create mask of frequencies (following np.fft.rfftfreq and np.fft.fftfreq docs)
        h, w = x.shape[-2:]
        freq_y = torch.cat([torch.arange((h - 1) // 2 + 1, dtype=torch.float32),
                            -torch.arange(h // 2, 0, -1, dtype=torch.float32)]) / h  # fftfreq
        freq_x = torch.arange(w // 2 + 1, dtype=torch.float32) / w  # rfftfreq
        yx_freq = torch.sqrt(freq_y[:, None] ** 2 + freq_x ** 2)

        # Create smoothing mask
        norm_freq = yx_freq * torch.sqrt(torch.tensor(2.0))  # 0-1
        mask = (1 - norm_freq) ** self.freq_exp

        # Smooth
        freq = torch.rfft(x, signal_ndim=2)  # same output as np.fft.rfft2
        mask = torch.as_tensor(mask, device=freq.device, dtype=freq.dtype).unsqueeze(-1)
        smooth = torch.irfft(freq * mask, signal_ndim=2, signal_sizes=x.shape[-2:])
        return smooth


class DivideByMeanOfAbsolute():
    """ Divides x by the mean of absolute x. """
    @varargin
    def __call__(self, x):
        return x / torch.abs(x).view(len(x), -1).mean(-1)


class MultiplyBy():
    """Multiply x by some constant.

    Arguments:
        const: Number x will be multiplied by
        decay_factor: Compute const every iteration as `const + decay_factor * (iteration
            - 1)`. Ignored if None.
    """
    def __init__(self, const, decay_factor=None, every_n_iterations=None):
        self.const = const
        self.decay_factor = decay_factor
        self.every_n_iterations = every_n_iterations

    @varargin
    def __call__(self, x, iteration=None):
        if self.decay_factor is None:
            const = self.const
        else:
            # const = self.const + self.decay_factor * (iteration - 1)
            const = self.const + self.decay_factor * ((iteration - 1) // self.every_n_iterations)
        return const * x

class Slicing():
    """
    Slice x by one certain index.
    """
    def __init__(self, idx):
        self.idx = idx

    @varargin
    def __call__(self, x, iteration=None):
        if type(x) == 'tuple':
            return x[self.idx]
        else:
            return x[:, self.idx]
    

########################### POST UPDATE OPERATIONS #######################################
class GaussianBlur():
    """ Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """
    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode='reflect'):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode

    @varargin
    def __call__(self, x, iteration=None):
        num_channels = x.shape[1]

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                         mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None],
                             groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1),
                             groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        return final_x


class ChangeStd():
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
    """
    def __init__(self, std):
        self.std = std

    @varargin
    def __call__(self, x):
        x_std = torch.std(x.view(len(x), -1), dim=-1)
        fixed_std = x * (self.std / (x_std + 1e-9)).view(len(x), *[1, ] * (x.dim() - 1))
        return fixed_std

class ChangeStats():
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
    """
    def __init__(self, std, mean):
        self.std = std
        self.mean = mean
        
    @varargin
    def __call__(self, x):
        x_std = torch.std(x, (-1, -2), keepdim=True)
        x_mean = torch.mean(x, (-1, -2), keepdim=True)
        fixed_im = (x - x_mean) * (self.std / (x_std + 1e-9)) + self.mean
        return fixed_im

class ChangeMaskStd():
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
    """
    def __init__(self, std, mask, fix_bg=False, bg=0):
        self.std = std
        self.mask = mask
        self.bg = bg

    @varargin
    def __call__(self, x):
        mask_mean = torch.sum(x * self.mask, (-1, -2), keepdim=True) / self.mask.sum()
        mask_std = torch.sqrt(torch.sum(((x - mask_mean) ** 2) * self.mask, (-1, -2), keepdim=True) / self.mask.sum())
        fixed_std = x * (self.std / (mask_std + 1e-9)).view(len(x), *[1, ] * (x.dim() - 1))
        if self.fix_bg:
            fixed_std = fixed_std * self.mask + self.bg * (1 - self.mask)
        return fixed_std

class ChangeMaskStats():
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
    """
    def __init__(self, std, mean, mask, fix_bg=False):
        self.std = std
        self.mask = mask
        self.mean = mean
        self.fix_bg = fix_bg

    @varargin
    def __call__(self, x):
        mask_mean = torch.sum(x * self.mask, (-1, -2), keepdim=True) / self.mask.sum()
        mask_std = torch.sqrt(torch.sum(((x - mask_mean) ** 2) * self.mask, (-1, -2), keepdim=True) / self.mask.sum())
        fixed_im = (x - mask_mean) * (self.std / (mask_std + 1e-9)).view(len(x), *[1, ] * (x.dim() - 1)) + self.mean
        if self.fix_bg:
            fixed_im = fixed_im * self.mask + self.mean * (1 - self.mask)
        return fixed_im

def get_mask_stats(image, mask):
    mask_mean = (image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
    mask_std = np.sqrt(np.sum(((image - mask_mean) ** 2) * mask, axis=(-1, -2), keepdims=True) / mask.sum())
    return mask_mean, mask_std

def standardize_image(image, target_mean, target_std, mask=None, mask_mean_subtraction=True, mask_image=True, match_stats='ff'):
    if match_stats == 'mask':
        assert mask is not None, 'Cannot match statistics within mask when mask is not provided!' 
        mean = (image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        std = np.sqrt(np.sum(((image - mean) ** 2) * mask, axis=(-1, -2), keepdims=True) / mask.sum())
        if mask_mean_subtraction:
            image = (image - mean) / (std + 1e-9) * target_std + target_mean
        else:
            image = image / (std + 1e-9) * target_std
        if mask is not None and mask_image:
            image = image * mask
    elif match_stats == 'ff':
        if mask_mean_subtraction:
            if mask is not None:
                mean = (image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
                image = image - mean
        if mask is not None and mask_image:
            image = image * mask
        image = (image - image.mean(axis=(-1, -2), keepdims=True)) / (image.std(axis=(-1, -2), keepdims=True) + 1e-9) * target_std + target_mean

    return image

def center_and_crop_image(image, x_offset, y_offset, output_size=(32, 32)):
    """ Center image to the closest integer and crop it

    Arguments:
        image (array): Original image
        x_offset (float): How far to the right (in x) is the image center.
        y_offset (float): How far to the bottom (in y) is the image center.
        output_size (tuple): How far to the right is the image center.

    Returns:
        Centered image after cropping to the desired output size.
    """
    
    x_offset = x_offset * int(output_size[-1] // 32)
    y_offset = y_offset * int(output_size[-2] // 32)
    
    top = (image.shape[0] - output_size[0]) / 2 + y_offset
    bottom = (image.shape[0] - output_size[0]) / 2 - y_offset
    left = (image.shape[1] - output_size[1]) / 2 + x_offset
    right = (image.shape[1] - output_size[1]) / 2 - x_offset
    top, bottom, left, right = (int(round(x)) for x in [top, bottom, left, right])

    # Pad image
    pad_amount = ((abs(top) if top < 0 else 0, abs(bottom) if bottom < 0 else 0),
                  (abs(left) if left < 0 else 0, abs(right) if right < 0 else 0))
    padded = np.pad(image, pad_amount, mode='edge')
    top, bottom, left, right = (np.clip(top, 0, None), np.clip(bottom, 0, None),
                                np.clip(left, 0, None), np.clip(right, 0, None))

    # Crop
    cropped = padded[top:padded.shape[0] - bottom, left:padded.shape[1] - right]
    if cropped.shape != output_size:
        raise ValueError('Something wrong with the cropping. This may fail for '
                         'images with odd dimensions.')

    return cropped

def create_whole_mei(crop, mask, center_x, center_y, output_size=(36, 64), normalize_crop=True):
    if len(crop.shape) == 2: 
        if normalize_crop:
            crop = (crop - crop.mean()) / (crop.std() + 1e-9)
        top = (crop.shape[0] - output_size[0]) / 2 - center_y
        bottom = (crop.shape[0] - output_size[0]) / 2 + center_y
        left = (crop.shape[1] - output_size[1]) / 2 - center_x
        right = (crop.shape[1] - output_size[1]) / 2 + center_x
        top, bottom, left, right = (int(round(x)) for x in [top, bottom, left, right])

        # Pad image
        pad_amount = ((abs(top) if top < 0 else 0, abs(bottom) if bottom < 0 else 0),
                      (abs(left) if left < 0 else 0, abs(right) if right < 0 else 0))
    elif len(crop.shape) == 3:
        if normalize_crop:
            # Error note: for group 222 with control_params=3, crop = (crop - crop.mean()) / crop.std() was used. This is wrong!
            mean, std = crop.mean(axis=(1, 2), keepdims=True), crop.std(axis=(1, 2), keepdims=True)
            crop = (crop - mean) / (std + 1e-9)
        top = (crop.shape[1] - output_size[1]) / 2 - center_y
        bottom = (crop.shape[1] - output_size[1]) / 2 + center_y
        left = (crop.shape[2] - output_size[2]) / 2 - center_x
        right = (crop.shape[2] - output_size[2]) / 2 + center_x
        top, bottom, left, right = (int(round(x)) for x in [top, bottom, left, right])

        # Pad image
        pad_amount = ((0, 0),
                      (abs(top) if top < 0 else 0, abs(bottom) if bottom < 0 else 0),
                      (abs(left) if left < 0 else 0, abs(right) if right < 0 else 0))

    padded = np.pad(crop, pad_amount, mode='constant', constant_values=0)
    top, bottom, left, right = (np.clip(top, 0, None), np.clip(bottom, 0, None),
                                np.clip(left, 0, None), np.clip(right, 0, None))
    
    # Crop
    if len(crop.shape) == 2:
        cropped = padded[top:padded.shape[0] - bottom, left:padded.shape[1] - right]
    elif len(crop.shape) == 3:
        cropped = padded[:, top:padded.shape[1] - bottom, left:padded.shape[2] - right]

    if cropped.shape != output_size:
        raise ValueError('Something wrong with the cropping. This may fail for images'
                         'with odd dimensions.')
    return cropped

def get_mei_sim(images, mei, mask=None):
    if mask is not None:
        images = images * mask
        mei = mei * mask
    images = images.reshape(len(images), -1)
    mei = mei.reshape(1, -1)
    sims = - np.linalg.norm(images - mei, axis=-1)
    return sims

def get_batch(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]