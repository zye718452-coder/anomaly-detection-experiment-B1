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
        # x [B, L, D]
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
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True,
                 use_b1=False, ma_window=5, alpha_base=1.2, alpha_range=0.3, alpha_hidden=-1):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # ===== B1 settings =====
        self.use_b1 = use_b1
        self.ma_window = ma_window
        self.alpha_base = alpha_base
        self.alpha_range = alpha_range

        hidden_dim = max(enc_in, 16) if alpha_hidden == -1 else alpha_hidden

        # 输入统计特征是 [mean_abs, max_abs]，所以维度是 2 * C
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * enc_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, enc_in)
        )

        # 可选：记录最近一次 alpha，后面调试或分析用
        self.last_alpha = None

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def moving_avg(self, x):
        """
        x: [B, L, C]
        return bg: [B, L, C]
        """
        pad = (self.ma_window - 1) // 2

        # [B, L, C] -> [B, C, L]
        x_t = x.transpose(1, 2)

        # replicate padding，避免边界太硬
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')

        # 按时间维做平均池化
        bg = F.avg_pool1d(x_pad, kernel_size=self.ma_window, stride=1)

        # [B, C, L] -> [B, L, C]
        bg = bg.transpose(1, 2)
        return bg

    def b1_enhance(self, x):
        """
        x: [B, L, C]
        """
        if not self.use_b1:
            return x

        # 1. background
        bg = self.moving_avg(x)

        # 2. residual
        residual = x - bg

        # 3. residual statistics: mean + max on |residual|
        mean_abs = residual.abs().mean(dim=1)            # [B, C]
        max_abs = residual.abs().max(dim=1).values       # [B, C]
        stat = torch.cat([mean_abs, max_abs], dim=-1)    # [B, 2C]

        # 4. alpha
        alpha_delta = self.alpha_mlp(stat)               # [B, C]
        alpha = self.alpha_base + self.alpha_range * torch.tanh(alpha_delta)  # [B, C]

        # 保存一份，便于后续分析
        self.last_alpha = alpha.unsqueeze(1)             # [B, 1, C]

        # 5. enhanced input
        x_enhanced = x + alpha.unsqueeze(1) * residual   # [B, L, C]

        return x_enhanced

    def forward(self, x):
        # ===== B1 front-end enhancement =====
        x = self.b1_enhance(x)

        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]