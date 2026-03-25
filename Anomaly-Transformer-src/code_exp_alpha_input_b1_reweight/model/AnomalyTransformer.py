import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
from .residual_enhance import B1ResidualEnhance


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, e0=None, return_gate=False):
        if return_gate:
            new_x, attn, mask, sigma, gate = self.attention(
                x, x, x, attn_mask=attn_mask, e0=e0, return_gate=True
            )
        else:
            new_x, attn, mask, sigma = self.attention(
                x, x, x, attn_mask=attn_mask, e0=e0
            )
            gate = None

        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if return_gate:
            return self.norm2(x + y), attn, mask, sigma, gate
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, e0=None):
        series_list = []
        prior_list = []
        sigma_list = []
        gate_list = []

        for i, attn_layer in enumerate(self.attn_layers):
            if i == len(self.attn_layers) - 1:
                x, series, prior, sigma, gate = attn_layer(
                    x, attn_mask=attn_mask, e0=e0, return_gate=True
                )
                gate_list.append(gate)
            else:
                x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)

            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, gate_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # B1 frontend
        self.residual_enhance = B1ResidualEnhance(
            c_in=enc_in,
            ma_kernel=5,
            alpha_hidden=128,
            alpha_base=1.2,
            alpha_range=0.6
        )

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # 这里不做 hidden reweight
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size,
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model,
                        n_heads,
                        layer_reweight=False,
                        c_in=None,
                        gamma=0.0,
                        dropout=dropout
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        # x: [B, L, C]
        x_enh, e0, aux = self.residual_enhance(x)

        enc_out = self.embedding(x_enh)
        enc_out, series, prior, sigmas, gates = self.encoder(enc_out, e0=None)

        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas, e0
        else:
            return enc_out