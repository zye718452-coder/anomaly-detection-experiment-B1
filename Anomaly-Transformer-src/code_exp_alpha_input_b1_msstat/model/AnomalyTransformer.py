import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding


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


class MovingAvgBackground(nn.Module):
    """
    Channel-wise moving average background extractor.
    Input:  x [B, L, C]
    Output: bg [B, L, C]
    """
    def __init__(self, kernel_size=5):
        super(MovingAvgBackground, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        # x: [B, L, C]
        x_t = x.transpose(1, 2)  # [B, C, L]

        if self.pad > 0:
            left = x_t[:, :, 0:1].repeat(1, 1, self.pad)
            right = x_t[:, :, -1:].repeat(1, 1, self.pad)
            x_pad = torch.cat([left, x_t, right], dim=-1)
        else:
            x_pad = x_t

        bg = self.avg(x_pad)     # [B, C, L]
        bg = bg.transpose(1, 2)  # [B, L, C]
        return bg


class MultiScaleAlphaGenerator(nn.Module):
    """
    B1 + multi-scale residual statistics

    Use residual statistics from multiple temporal scales to generate
    sample-dependent channel-wise alpha.

    Statistics used:
        for each scale s in {5,17,33}
            mean(abs(r_s)), max(abs(r_s))

    So res_stat dim = 6
    """
    def __init__(self,
                 alpha_base=1.2,
                 alpha_scale=0.3,
                 hidden_dim=16):
        super(MultiScaleAlphaGenerator, self).__init__()
        self.alpha_base = alpha_base
        self.alpha_scale = alpha_scale
        self.hidden_dim = hidden_dim

        # input dim = 6: [mean_5, max_5, mean_17, max_17, mean_33, max_33]
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def extract_scale_stat(self, residual):
        """
        residual: [B, L, C]
        return:   [B, C, 2] = [mean(abs(res)), max(abs(res))]
        """
        abs_res = residual.abs()
        res_mean = abs_res.mean(dim=1)      # [B, C]
        res_max = abs_res.max(dim=1)[0]     # [B, C]
        res_stat = torch.stack([res_mean, res_max], dim=-1)  # [B, C, 2]
        return res_stat

    def forward(self, r5, r17, r33):
        """
        r5, r17, r33: [B, L, C]
        return:
            alpha:    [B, 1, C]
            res_stat: [B, C, 6]
        """
        stat5 = self.extract_scale_stat(r5)      # [B, C, 2]
        stat17 = self.extract_scale_stat(r17)    # [B, C, 2]
        stat33 = self.extract_scale_stat(r33)    # [B, C, 2]

        res_stat = torch.cat([stat5, stat17, stat33], dim=-1)  # [B, C, 6]

        h = F.relu(self.fc1(res_stat))           # [B, C, H]
        delta = self.alpha_scale * torch.tanh(self.fc2(h))  # [B, C, 1]

        alpha = self.alpha_base + delta          # [B, C, 1]
        alpha = alpha.transpose(1, 2)            # [B, 1, C]

        return alpha, res_stat


class AnomalyTransformer(nn.Module):
    def __init__(self,
                 win_size,
                 enc_in,
                 c_out,
                 d_model=512,
                 n_heads=8,
                 e_layers=3,
                 d_ff=512,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=True,
                 bg_kernel_size=5,
                 alpha_base=1.2,
                 alpha_scale=0.3,
                 alpha_hidden_dim=16):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # main enhancement scale: keep B1 main branch unchanged
        self.bg_extractor_5 = MovingAvgBackground(kernel_size=bg_kernel_size)

        # auxiliary scales only for alpha generation
        self.bg_extractor_17 = MovingAvgBackground(kernel_size=17)
        self.bg_extractor_33 = MovingAvgBackground(kernel_size=33)

        self.alpha_generator = MultiScaleAlphaGenerator(
            alpha_base=alpha_base,
            alpha_scale=alpha_scale,
            hidden_dim=alpha_hidden_dim
        )

        self.last_alpha = None
        self.last_res_stat = None

        # original backbone unchanged
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

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
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def front_end_enhance(self, x):
        """
        x: [B, L, C]

        Main enhancement still uses single-scale residual r5.
        Multi-scale residual stats are only used to estimate alpha.
        """
        bg5 = self.bg_extractor_5(x)
        bg17 = self.bg_extractor_17(x)
        bg33 = self.bg_extractor_33(x)

        r5 = x - bg5
        r17 = x - bg17
        r33 = x - bg33

        alpha, res_stat = self.alpha_generator(r5, r17, r33)   # alpha: [B, 1, C]

        x_enhanced = x + alpha * r5

        self.last_alpha = alpha.squeeze(1).detach()   # [B, C]
        self.last_res_stat = res_stat.detach()        # [B, C, 6]

        aux_info = {
            "bg5": bg5,
            "bg17": bg17,
            "bg33": bg33,
            "r5": r5,
            "r17": r17,
            "r33": r33,
            "alpha": alpha,
            "res_stat": res_stat
        }
        return x_enhanced, aux_info

    def forward(self, x):
        # keep original 4-return interface for solver compatibility
        x, _ = self.front_end_enhance(x)

        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out