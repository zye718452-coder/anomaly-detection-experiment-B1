import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverageBackground(nn.Module):
    def __init__(self, kernel_size=5):
        super(MovingAverageBackground, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        """
        x: [B, L, C]
        return:
            bg: [B, L, C]
        """
        xt = x.transpose(1, 2)  # [B, C, L]
        bg = F.avg_pool1d(
            F.pad(xt, (self.pad, self.pad), mode='replicate'),
            kernel_size=self.kernel_size,
            stride=1
        )
        bg = bg.transpose(1, 2)  # [B, L, C]
        return bg


class EvidenceTarget(nn.Module):
    """
    Build residual evidence target:
        e = x - MA(x)
    """
    def __init__(self, ma_kernel=5):
        super(EvidenceTarget, self).__init__()
        self.background = MovingAverageBackground(ma_kernel)

    def forward(self, x):
        """
        x: [B, L, C]
        returns:
            e:  [B, L, C]
            bg: [B, L, C]
        """
        bg = self.background(x)
        e = (x - bg).abs()
        return e, bg