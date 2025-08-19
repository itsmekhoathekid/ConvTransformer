import torch
from torch import nn
from .modules import FeedForwardBlock, ResidualConnection, PositionalEncoding, ConvDec, VGGFrontEnd
from .attention import MultiHeadAttentionBlock

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttentionBlock(d_model, h, p_dropout)
        self.feed_forward = FeedForwardBlock(d_model, ff_size,  p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.residual_connections = nn.ModuleList(
            ResidualConnection(d_model, p_dropout) for _ in range(2)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.residual_connections[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residual_connections[1](x, lambda x : self.feed_forward(x))
        self.dropout(x)
        return x


class VGGTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=d_model)
        self.pe = PositionalEncoding(d_model)
        

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.frontend = VGGFrontEnd(
            num_blocks = 2,
            in_channel=1,
            out_channels=[32, 64],
            conv_kernel_sizes=[3, 3],
            pooling_kernel_sizes=[2, 2],
            num_conv_layers=[2, 2],
            layer_norms=[True, True],
            input_dim=input_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        # print("x shape before frontend:", x.shape)  # [batch, 1, time, features]
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        # print("x shape after frontend:", x.shape)
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]


        out = self.linear(x)
        out = self.pe(out)
        mask_atten = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            out = layer(out, mask_atten)
        
        return out, mask
