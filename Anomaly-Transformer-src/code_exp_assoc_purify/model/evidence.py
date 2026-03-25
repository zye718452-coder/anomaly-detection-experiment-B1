import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverageBackground(nn.Module):
    """
    Raw-input moving average background extractor.
    Input/Output shape: [B, L, C]
    """
    def __init__(self, kernel_size=5):
        super(MovingAverageBackground, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        xt = x.transpose(1, 2)
        bg = F.avg_pool1d(
            F.pad(xt, (self.pad, self.pad), mode='replicate'),
            kernel_size=self.kernel_size,
            stride=1
        )
        bg = bg.transpose(1, 2)
        return bg


class EvidenceScorer(nn.Module):
    """
    Build token-level evidence score from raw residual:
        b = MA(x)
        e = x - b
        token_energy = mean(|e| over channels)
        local mean / local var on token_energy
        small Conv1d -> score s in [0,1]
    """
    def __init__(self, c_in, ma_kernel=5, local_kernel=5, hidden_dim=16):
        super(EvidenceScorer, self).__init__()
        assert local_kernel % 2 == 1, "local_kernel must be odd"

        self.background = MovingAverageBackground(ma_kernel)
        self.local_kernel = local_kernel
        self.local_pad = local_kernel // 2

        self.scorer = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )

        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def _local_stats(self, z):
        # z: [B, 1, L]
        z_pad = F.pad(z, (self.local_pad, self.local_pad), mode='replicate')
        mean = F.avg_pool1d(z_pad, kernel_size=self.local_kernel, stride=1)

        z2_pad = F.pad(z * z, (self.local_pad, self.local_pad), mode='replicate')
        mean2 = F.avg_pool1d(z2_pad, kernel_size=self.local_kernel, stride=1)

        var = torch.clamp(mean2 - mean * mean, min=0.0)
        return mean, var

    def forward(self, x):
        """
        x: [B, L, C]

        returns:
            s_raw: [B, L]
            s:     [B, L] in [0,1]
            e:     [B, L, C]
            b:     [B, L, C]
        """
        b = self.background(x)
        e = x - b

        token_energy = e.abs().mean(dim=-1, keepdim=True)   # [B, L, 1]
        token_energy_t = token_energy.transpose(1, 2)       # [B, 1, L]

        local_mean, local_var = self._local_stats(token_energy_t)

        feat = torch.cat([token_energy_t, local_mean, local_var], dim=1)  # [B, 3, L]
        s_raw = self.scorer(feat).squeeze(1)  # [B, L]
        s = torch.sigmoid(s_raw)

        return s_raw, s, e, b