import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
from .evidence import EvidenceScorer


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

    def forward(self, x, attn_mask=None, attn_bias=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            attn_bias=attn_bias
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

    def forward(self, x, attn_mask=None, attn_bias=None):
        series_list = []
        prior_list = []
        sigma_list = []

        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(
        self,
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
        evidence_bias=True,
        evidence_lambda=2.0,
        evidence_ma_kernel=5,
        evidence_local_kernel=5,
        evidence_hidden_dim=16,
        evidence_share_across_heads=True
    ):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.win_size = win_size
        self.n_heads = n_heads

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
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_evidence_bias = evidence_bias
        self.evidence_lambda = evidence_lambda
        self.evidence_share_across_heads = evidence_share_across_heads

        if self.use_evidence_bias:
            self.evidence_scorer = EvidenceScorer(
                c_in=enc_in,
                ma_kernel=evidence_ma_kernel,
                local_kernel=evidence_local_kernel,
                hidden_dim=evidence_hidden_dim
            )

    def build_attention_bias(self, s):
        """
        s: [B, L], range [0,1]

        column-wise source suppression:
            bias[b, h, i, j] = -lambda * s[b, j]

        meaning:
            suspicious token j is harder to be attended by any query i
        """
        col_bias = -self.evidence_lambda * s.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]

        if self.evidence_share_across_heads:
            return col_bias
        else:
            B, _, _, L = col_bias.shape
            return col_bias.repeat(1, self.n_heads, L, 1)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        attn_bias = None
        aux = {}

        if self.use_evidence_bias:
            s_raw, s, e, b = self.evidence_scorer(x)
            attn_bias = self.build_attention_bias(s)

            aux["evidence_score_raw"] = s_raw
            aux["evidence_score"] = s
            aux["evidence_residual"] = e
            aux["evidence_background"] = b

        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out, attn_mask=None, attn_bias=attn_bias)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas, aux
        else:
            return enc_out, aux