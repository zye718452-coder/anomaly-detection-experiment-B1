import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool),
                diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class EvidenceProjector(nn.Module):
    """
    E0: [B, L, c_in] -> E: [B, L, d_model]
    更稳版本：LayerNorm + 2层MLP
    """
    def __init__(self, c_in, d_model, hidden_dim=None, dropout=0.0):
        super(EvidenceProjector, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(max(d_model, 128), 512)

        self.norm = nn.LayerNorm(c_in)
        self.fc1 = nn.Linear(c_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, e0):
        x = self.norm(e0)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TokenEvidenceGate(nn.Module):
    """
    E: [B, L, d_model] -> g: [B, L, 1]
    更稳版本：LayerNorm + 2层MLP
    """
    def __init__(self, d_model, hidden_dim=None, dropout=0.0):
        super(TokenEvidenceGate, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(max(d_model // 2, 128), 512)

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e):
        x = self.norm(e)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        g = self.sigmoid(x)
        return g


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size

        distances = torch.zeros((win_size, win_size)).float()
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        self.register_buffer("distances", distances)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # [B, H, L]
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(torch.tensor(3.0, device=sigma.device), sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # [B, H, L, L]

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1).to(sigma.device)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(
            -prior ** 2 / 2 / (sigma ** 2)
        )

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None, None, None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 layer_reweight=False, c_in=None, gamma=0.1, dropout=0.0):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.layer_reweight = layer_reweight
        self.gamma = gamma

        if self.layer_reweight:
            assert c_in is not None
            self.evidence_projector = EvidenceProjector(
                c_in, d_model, hidden_dim=None, dropout=dropout
            )
            self.token_gate = TokenEvidenceGate(
                d_model, hidden_dim=None, dropout=dropout
            )

    def forward(self, queries, keys, values, attn_mask, e0=None, return_gate=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask
        )

        out = out.view(B, L, -1)
        out = self.out_projection(out)  # [B, L, d_model]

        gate = None
        if self.layer_reweight:
            if e0 is None:
                raise ValueError("layer_reweight=True 时必须传入 e0")
            e = self.evidence_projector(e0)               # [B, L, d_model]
            gate = self.token_gate(e)                     # [B, L, 1]
            gate = gate - gate.mean(dim=1, keepdim=True) # zero-sum modulation
            out = out * (1.0 + self.gamma * gate)

        if return_gate:
            return out, series, prior, sigma, gate
        return out, series, prior, sigma