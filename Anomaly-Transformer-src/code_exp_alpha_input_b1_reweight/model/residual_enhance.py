import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgBackground(nn.Module):
    def __init__(self, kernel_size=5):
        super(MovingAvgBackground, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        # x: [B, L, C]
        x_t = x.transpose(1, 2)  # [B, C, L]
        x_t = F.pad(x_t, (self.pad, self.pad), mode='replicate')
        bg = self.avg(x_t)
        bg = bg.transpose(1, 2)  # [B, L, C]
        return bg


class ResidualAlphaNet(nn.Module):
    """
    B1: sample-dependent, channel-wise alpha
    input: residual r [B, L, C]
    output: alpha [B, 1, C]
    """
    def __init__(self, c_in, hidden_dim=128, alpha_base=1.2, alpha_range=0.6):
        super(ResidualAlphaNet, self).__init__()
        self.c_in = c_in
        self.alpha_base = alpha_base
        self.alpha_range = alpha_range

        stat_dim = c_in * 2  # mean + max
        hidden_dim = min(hidden_dim, max(32, stat_dim))

        self.mlp = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c_in)
        )

    def forward(self, r):
        # r: [B, L, C]
        r_mean = r.mean(dim=1)             # [B, C]
        r_max = r.abs().max(dim=1).values  # [B, C]
        stat = torch.cat([r_mean, r_max], dim=-1)  # [B, 2C]

        delta = torch.sigmoid(self.mlp(stat))      # [B, C]
        alpha = self.alpha_base + self.alpha_range * (delta - 0.5)
        alpha = alpha.unsqueeze(1)                 # [B, 1, C]
        return alpha


class B1ResidualEnhance(nn.Module):
    """
    输出:
        x_enh: [B, L, C]
        e0:    [B, L, C]   evidence for reweight
        aux:   dict
    """
    def __init__(self, c_in, ma_kernel=5, alpha_hidden=128,
                 alpha_base=1.2, alpha_range=0.6):
        super(B1ResidualEnhance, self).__init__()
        self.bg_estimator = MovingAvgBackground(kernel_size=ma_kernel)
        self.alpha_net = ResidualAlphaNet(
            c_in=c_in,
            hidden_dim=alpha_hidden,
            alpha_base=alpha_base,
            alpha_range=alpha_range
        )

    def forward(self, x):
        # x: [B, L, C]
        bg = self.bg_estimator(x)
        r = x - bg
        alpha = self.alpha_net(r)      # [B,1,C]
        e0 = alpha * r                 # [B,L,C]
        x_enh = x + e0                 # [B,L,C]

        aux = {
            'bg': bg,
            'residual': r,
            'alpha': alpha
        }
        return x_enh, e0, aux