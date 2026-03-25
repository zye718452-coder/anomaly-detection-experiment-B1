import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage1D(nn.Module):
    """
    Moving average over temporal dimension for input shape [B, L, C].
    """
    def __init__(self, kernel_size: int = 5):
        super(MovingAverage1D, self).__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, \
            "kernel_size must be a positive odd integer."
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, L, C]
        """
        x_t = x.transpose(1, 2)  # [B, C, L]
        x_pad = F.pad(x_t, (self.pad, self.pad), mode='replicate')
        bg = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return bg.transpose(1, 2)  # [B, L, C]


class EvidenceScoreFusion(nn.Module):
    """
    Evidence-guided gated score fusion using mean+max residual statistics.

    Steps:
    1) background = moving average(x)
    2) residual = x - background
    3) evidence raw score = mean(abs(residual), dim=channel) + max(abs(residual), dim=channel)
    4) temporal z-score normalization per sample
    5) one-sided positive evidence = ReLU(z-score)
    6) gate = 1 only on high main-score positions within each sample
    7) final score = main_score + lambda_evi * gate * positive_evidence

    Intuition:
    - main_score remains dominant
    - evidence only boosts already suspicious positions
    - avoids global score distribution distortion
    """
    def __init__(self, ma_window=5, eps=1e-4, lambda_evi=0.005, gate_quantile=0.90):
        super(EvidenceScoreFusion, self).__init__()
        self.ma = MovingAverage1D(ma_window)
        self.eps = eps
        self.lambda_evi = lambda_evi
        self.gate_quantile = gate_quantile

    @torch.no_grad()
    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: residual [B, L, C]
        """
        bg = self.ma(x)
        residual = x - bg
        return residual

    @torch.no_grad()
    def compute_evidence_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: evidence score [B, L]
        """
        residual = self.compute_residual(x)
        abs_res = torch.abs(residual)            # [B, L, C]

        evi_mean = torch.mean(abs_res, dim=-1)   # [B, L]
        evi_max = torch.max(abs_res, dim=-1)[0]  # [B, L]

        evi = evi_mean + evi_max
        return evi

    @torch.no_grad()
    def normalize_over_time(self, score: torch.Tensor) -> torch.Tensor:
        """
        score: [B, L]
        return: [B, L]
        """
        mean = torch.mean(score, dim=1, keepdim=True)
        std = torch.std(score, dim=1, keepdim=True, unbiased=False)
        score_norm = (score - mean) / (std + self.eps)
        return score_norm

    @torch.no_grad()
    def positive_evidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, L]
        """
        evi = self.compute_evidence_score(x)
        evi_norm = self.normalize_over_time(evi)
        evi_pos = F.relu(evi_norm)
        return evi_pos

    @torch.no_grad()
    def make_gate(self, main_score: torch.Tensor) -> torch.Tensor:
        """
        main_score: [B, L]
        return: gate [B, L]

        Gate is 1 only for positions whose main_score is above the
        sample-wise quantile threshold.
        """
        q = torch.quantile(main_score, self.gate_quantile, dim=1, keepdim=True)
        gate = (main_score > q).float()
        return gate

    @torch.no_grad()
    def fuse(self, main_score: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        main_score: [B, L]
        x: [B, L, C]
        return: [B, L]
        """
        evi_pos = self.positive_evidence(x)
        gate = self.make_gate(main_score)
        fused_score = main_score + self.lambda_evi * gate * evi_pos
        return fused_score