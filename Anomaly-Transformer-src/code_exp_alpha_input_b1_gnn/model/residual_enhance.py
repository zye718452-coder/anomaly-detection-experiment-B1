# model/residual_enhance.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgBackground(nn.Module):
    """
    Moving average background extractor for inputs of shape [B, L, C].

    We smooth along the temporal dimension L independently for each channel.
    """
    def __init__(self, kernel_size: int):
        super(MovingAvgBackground, self).__init__()
        assert kernel_size >= 1, "kernel_size must be >= 1"
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return bg: [B, L, C]
        """
        if self.kernel_size <= 1:
            return x

        # [B, L, C] -> [B, C, L]
        x_t = x.permute(0, 2, 1)

        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left

        # replicate padding to avoid shrinking at boundaries
        x_pad = F.pad(x_t, (pad_left, pad_right), mode='replicate')

        # avg_pool1d over temporal dimension
        bg = F.avg_pool1d(
            x_pad,
            kernel_size=self.kernel_size,
            stride=1
        )

        # [B, C, L] -> [B, L, C]
        bg = bg.permute(0, 2, 1)
        return bg


class LearnableResidualEnhancer(nn.Module):
    """
    x_enhanced = x + alpha * (x - bg)

    alpha is channel-wise learnable, parameterized as:
        alpha = alpha_max * sigmoid(alpha_logits)
    so alpha is constrained in (0, alpha_max).
    """
    def __init__(
        self,
        c_in: int,
        ma_kernel_size: int = 25,
        alpha_init: float = 1.2,
        alpha_max: float = 2.0
    ):
        super(LearnableResidualEnhancer, self).__init__()

        assert c_in > 0, "c_in must be positive"
        assert alpha_max > 0, "alpha_max must be positive"
        assert 0.0 < alpha_init < alpha_max, \
            "alpha_init must be in (0, alpha_max)"

        self.c_in = c_in
        self.alpha_max = alpha_max
        self.background = MovingAvgBackground(kernel_size=ma_kernel_size)

        # inverse sigmoid for initialization:
        # alpha_init = alpha_max * sigmoid(logits_init)
        # => sigmoid(logits_init) = alpha_init / alpha_max
        ratio = alpha_init / alpha_max
        logits_init = math.log(ratio / (1.0 - ratio))

        self.alpha_logits = nn.Parameter(
            torch.full((c_in,), float(logits_init))
        )

    def get_alpha(self) -> torch.Tensor:
        """
        return alpha of shape [C]
        """
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logits)
        return alpha

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        returns:
            x_enhanced: [B, L, C]
            bg:         [B, L, C]
            residual:   [B, L, C]
            alpha:      [C]
        """
        bg = self.background(x)
        residual = x - bg

        alpha = self.get_alpha()              # [C]
        alpha_bc = alpha.view(1, 1, -1)       # [1, 1, C]

        x_enhanced = x + alpha_bc * residual
        return x_enhanced, bg, residual, alpha