import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from einops import rearrange

default_model_urls = {
    'resnet50-koniq': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/HyperIQA-resnet50-koniq10k-c96c41b1.pth',
    'koniq10k': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CNNIQA_koniq10k-e6f14c91.pth',
}


class SVDLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        scale: float = 1.0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)        
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()
    
    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True    

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        return F.linear(x, weight_updated, bias=self.bias)

class SVDConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        
        
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1), h=self.weight.size(2), w=self.weight.size(3))
        return F.conv2d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups)


@ARCH_REGISTRY.register()
class myiqa2(nn.Module):
    r"""myiqa model.
    Args:
        - ker_size (int): Kernel size.
        - n_kers (int): Number of kernals.
        - n1_nodes (int): Number of n1 nodes.
        - n2_nodes (int): Number of n2 nodes.
        - pretrained_model_path (String): Pretrained model path.

    """

    def __init__(
        self,
        ker_size=7,
        n_kers=50,
        n1_nodes=800,
        n2_nodes=800,
        pretrained='koniq10k',
        pretrained_model_path=None,
    ):
        super(myiqa2, self).__init__()

        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

        if pretrained_model_path is None and pretrained is not None:
            pretrained_model_path = default_model_urls[pretrained]

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')
            
        self.svdconv1 = SVDConv2d(3,n_kers,ker_size)
        #self.svdfc1 = SVDLinear(2* n_kers,n1_nodes)
        #self.svdfc2 = SVDLinear(n1_nodes, n2_nodes)
        #self.svdfc3 = SVDLinear(n2_nodes, 1)
        
        self.svdconv1.weight.data = self.conv1.weight.data
        #self.svdfc1.weight.data = self.fc1.weight.data
        #self.svdfc2.weight.data = self.fc2.weight.data
        #self.svdfc2.weight.data = self.fc2.weight.data


    def forward(self, x):
        r"""Compute IQA using myiqa model.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of CNNIQA model.

        """
        h = self.svdconv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q
