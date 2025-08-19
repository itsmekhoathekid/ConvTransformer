from __future__ import absolute_import, division, print_function, unicode_literals

from collections.abc import Iterable
from itertools import repeat

import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Type, List





class ConvDecBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x.float())  # (batch, out_channels, new_seq_len)
        x = x.transpose(1, 2)  # (batch, new_seq_len, out_channels)
        x = self.norm(x)  # (batch, new_seq_len, out_channels)
        x = self.relu(x)  # (batch, new_seq_len, out_channels)
        x = self.dropout(x)  # (batch, new_seq_len, out_channels)
        x = x.transpose(1, 2)
        return x


class ConvDec(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_sizes, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            conv_block = ConvDecBlock(
                in_channels, 
                out_channels[i], 
                kernel_sizes[i], 
                dropout)
            blocks.append(conv_block)
            in_channels = out_channels[i]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, in_channels)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (batch, seq_len, out_channels)
        
        return x



def _pair(v):
    if isinstance(v, Iterable):
        assert len(v) == 2, "len(v) != 2"
        return v
    return tuple(repeat(v, 2))


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    # N x C x H x W
    # N: sample_bsz, C: sample_inchannel, H: sample_seq_len, W: input_dim
    x = conv_op(x)
    # N x C x H x W
    x = x.transpose(1, 2)
    # N x H x C x W
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    # bsz: N, seq: H, CxW the rest
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

import torch.nn.functional as F

def pool_time_mask(mask, layer: nn.MaxPool2d, T):
    # mask: [B, T] (1 = valid, 0 = pad)
    # Lấy tham số theo trục time (dim=2 của input BxCxTxF)
    k_t = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
    s_t = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
    p_t = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
    d_t = layer.dilation if isinstance(layer.dilation, int) else layer.dilation[0]
    ceil = layer.ceil_mode

    # Dùng max_pool1d để OR các bước thời gian trong mỗi cửa sổ
    # [B, T] -> [B, 1, T] để pool1d
    m = mask.unsqueeze(1).float()
    m_pooled = F.max_pool1d(m, kernel_size=k_t, stride=s_t, padding=p_t,
                            dilation=d_t, ceil_mode=ceil)
    new_mask = (m_pooled.squeeze(1) > 0.5).to(mask.dtype)  # [B, T_out]

    # T_out tính bởi công thức của pooling (ceil_mode được PyTorch xử lý sẵn)
    T_out = new_mask.size(1)
    return new_mask, T_out

class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf

    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        padding: implicit paddings on both sides of the input.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False

    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        pooling_kernel_size,
        num_conv_layers,
        input_dim,
        conv_stride=1,
        padding=None,
        layer_norm=False,
    ):
        assert (
            input_dim is not None
        ), "Need input_dim for LayerNorm and infer_conv_output_dim"
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels # 64
        self.out_channels = out_channels # 3
        self.conv_kernel_size = _pair(conv_kernel_size) # 2 
        self.pooling_kernel_size = _pair(pooling_kernel_size) # 2 
        self.num_conv_layers = num_conv_layers 
        self.padding = (
            tuple(e // 2 for e in self.conv_kernel_size)
            if padding is None
            else _pair(padding)
        )
        self.conv_stride = _pair(conv_stride)

        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else out_channels,
                out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.padding,
            )
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(
                    conv_op, input_dim, in_channels if layer == 0 else out_channels
                )
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, input_dim, out_channels
            )

    def forward(self, x, mask):
        B, C, T, Fdim = x.shape  # x: [B, C, T, F]

        for layer in self.layers:
            x = layer(x)

            if isinstance(layer, nn.Conv2d):
                # cập nhật T theo conv (như bạn đang làm)
                k = layer.kernel_size[0]; s = layer.stride[0]
                d = layer.dilation[0]; p = layer.padding[0]
                out_T = (T + 2*p - d*(k - 1) - 1) // s + 1

                # cách 1: giữ logic calc_data_len của bạn
                pad_len = T - mask.sum(dim=1)
                data_len = mask.sum(dim=1)
                new_len = calc_data_len(
                    result_len=out_T, pad_len=pad_len, data_len=data_len,
                    kernel_size=k, stride=s,
                )
                mask = get_mask_from_lens(new_len, out_T)
                T = out_T

            elif isinstance(layer, nn.MaxPool2d):
                # cập nhật mask theo time-pooling (OR các bước trong mỗi cửa sổ)
                mask, T = pool_time_mask(mask, layer, T)

        return x, mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)

        x = x + pe
        return x


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return self.norm(x + self.dropout(sublayer(x)))

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class VGGFrontEnd(nn.Module):
    def __init__(self, num_blocks, in_channel, out_channels, conv_kernel_sizes, pooling_kernel_sizes, num_conv_layers, layer_norms, input_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                VGGBlock(
                    in_channels=in_channel if i == 0 else out_channels[i - 1],
                    out_channels=out_channels[i],
                    conv_kernel_size=conv_kernel_sizes,
                    pooling_kernel_size=pooling_kernel_sizes[i],
                    num_conv_layers=num_conv_layers[i],
                    input_dim=input_dim,  
                    conv_stride=1,
                    padding=None,
                    layer_norm=layer_norms[i]
                )
                
            )
            input_dim = self.blocks[-1].output_dim
    
    def forward(self, x, mask):
        for conv_layer in self.blocks:
            x, mask = conv_layer(x.float(), mask)
        return x, mask


class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch ,seqlen, d_model -> batch, seqlen, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)