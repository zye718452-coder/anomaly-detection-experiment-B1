import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


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

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0,
                 activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.win_size = win_size

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        # ===== multi-scale moving average background extractors =====
        self.avg_pool_1 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.avg_pool_2 = nn.AvgPool1d(kernel_size=17, stride=1, padding=8)
        self.avg_pool_3 = nn.AvgPool1d(kernel_size=33, stride=1, padding=16)

        # fixed residual enhancement strength
        self.res_scale = 1.2

    def decompose(self, x):
        """
        x: [B, L, C]
        bg: multi-scale moving-average background
        res: residual = x - bg
        """
        x_t = x.transpose(1, 2)   # [B, C, L]

        bg1_t = self.avg_pool_1(x_t)   # [B, C, L]
        bg2_t = self.avg_pool_2(x_t)   # [B, C, L]
        bg3_t = self.avg_pool_3(x_t)   # [B, C, L]

        # equal-weight multi-scale fusion
        bg_t = (bg1_t + bg2_t + bg3_t) / 3.0

        bg = bg_t.transpose(1, 2)      # [B, L, C]
        res = x - bg
        return res, bg

    def forward(self, x):
        # multi-scale residual-enhanced input
        x_res, x_bg = self.decompose(x)
        x_enhanced = x + self.res_scale * x_res

        enc_out = self.embedding(x_enhanced)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out