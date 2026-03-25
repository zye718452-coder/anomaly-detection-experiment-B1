import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cuda"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, window_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False, n_heads=8):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = window_size
        self.n_heads = n_heads

        distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                distances[i, j] = abs(i - j)
        self.register_buffer("distances", distances)

        # per-head gate
        self.head_gate_logit = nn.Parameter(torch.zeros(n_heads))
        self.last_gate = None

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        if H != self.n_heads:
            raise ValueError(f"Expected {self.n_heads} heads, but got {H} heads.")

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = scale * scores

        # series attention
        series = self.dropout(torch.softmax(attn, dim=-1))

        # prior
        sigma = sigma.transpose(1, 2)  # B L H -> B H L
        window_size = self.window_size

        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1
        ).to(sigma.device)

        prior = 1.0 / (sqrt(2 * np.pi) * sigma) * torch.exp(
            -prior ** 2 / (2 * (sigma ** 2))
        )

        # normalize prior
        prior_norm = prior / (torch.sum(prior, dim=-1, keepdim=True) + 1e-8)

        # per-head gate: [H] -> [1, H, 1, 1]
        gate = torch.sigmoid(self.head_gate_logit).view(1, H, 1, 1)
        self.last_gate = torch.sigmoid(self.head_gate_logit).detach().cpu().numpy()

        fused_attn = gate * series + (1.0 - gate) * prior_norm
        fused_attn = fused_attn / (torch.sum(fused_attn, dim=-1, keepdim=True) + 1e-8)

        V = torch.einsum("bhls,bshd->blhd", fused_attn, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(queries.reshape(B, L, -1)).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma