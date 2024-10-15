import torch
import torch.nn as nn
import numpy as np


from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)




class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)




class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        N, _, H, W = x.size()
        std = torch.std(x, dim=0, keepdim=True) # (1, C, H, W)
        std_mean = torch.mean(std, dim=(1,2,3), keepdim=True).expand(N, -1, H, W)
        # std_mean = torch.mean(std, dim=1, keepdim=True).expand(x.size(0), -1, -1, -1)
        return torch.cat([x, std_mean], dim=1) # (N, C+1, H, W)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1,
                    lrelu=True, weight_norm=False, snorm=True, equalized=True):
        super(Conv2d, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups
        )
        if weight_norm:
            nn.utils.weight_norm(self.conv)
        
        self.lrelu = LeakyReLU() if lrelu else None
        self.normalize = SpectralNorm(self.conv) if snorm else None
        self.equalized = equalized
        if equalized:
            self.conv.weight.data.normal_(0, 1)
            fan_in = np.prod(self.conv.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.conv.bias.data.fill_(0.)
        
    def forward(self, x): 
        y = self.conv(x)
        y = y*self.he_constant if self.equalized else y
        y = self.lrelu(y) if self.lrelu is not None else y
        y = self.normalize(y) if self.normalize is not None else y
        return y

class Linear(nn.Module):
    def __init__(self, in_dims, out_dims, 
                weight_norm=False, equalized=True):
        super(Linear, self).__init__()
        
        self.linear = nn.Linear(in_dims, out_dims)
        if weight_norm:
            nn.utils.weight_norm(self.linear)
        self.equalized = equalized
        if equalized:
            self.linear.weight.data.normal_(0, 1)
            fan_in = np.prod(self.linear.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.linear.bias.data.fill_(0.)

    def forward(self, x): 
        y = self.linear(x)
        y = y*self.he_constant if self.equalized else y
        return y
