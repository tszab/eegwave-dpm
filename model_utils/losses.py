
# Python imports
import torch
from torch import nn
from typing import Tuple
import numpy as np


class LogL1Loss(nn.Module):
    def __init__(self):
        super(LogL1Loss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (predictions - targets).abs().mean(dim=-1).clamp(min=1e-20).log().mean()


class KLDivergence(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(KLDivergence, self).__init__()
        self.reduction = reduction

    def forward(
            self,
            means: Tuple[torch.Tensor, torch.Tensor],
            vars: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        divergence = 0.5 * (-1. + vars[1] - vars[0] + torch.exp(vars[0] - vars[1]) +
                            ((means[0] - means[1]) ** 2) * torch.exp(-vars[1]))
        divergence = divergence.mean(dim=list(range(1, len(divergence.shape))))

        if self.reduction == 'mean':
            return divergence.mean(dim=0)
        elif self.reduction == 'sum':
            return divergence.sum(dim=0)
        else:
            return divergence


class DiscGaussianNLL(nn.Module):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    def __init__(self, reduction: str = 'mean'):
        super(DiscGaussianNLL, self).__init__()
        self.reduction = reduction

    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x, *, means, log_scales):
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 1.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 1.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape

        if self.reduction == 'mean':
            return log_probs.mean()
        elif self.reduction == 'sum':
            return log_probs.sum()
        else:
            return log_probs
