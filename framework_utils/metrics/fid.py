"""
Originally from https://github.com/kahartma/eeggan.
"""

# Python imports
import torch
from torch import nn


class FrechetInceptionDistance(nn.Module):
    def __init__(self, model: nn.Module):
        super(FrechetInceptionDistance, self).__init__()
        self.model = model
        self.distances = []

    def reset(self) -> None:
        self.distances = []

    @staticmethod
    def calculate_activation_statistics(act):
        with torch.no_grad():
            act = act.reshape(act.shape[0], -1)
            fact = act.shape[0] - 1
            mu = torch.mean(act, dim=0, keepdim=True)
            act = act - mu.expand_as(act)
            sigma = act.t().mm(act) / fact
            return mu, sigma

    @staticmethod
    def sqrtm_newton(A: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            numIters = 20
            batchSize = A.shape[0]
            dim = A.shape[1]
            normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
            Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
            I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
            Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
            for i in range(numIters):
                T = 0.5 * (3.0 * I - Z.bmm(Y))
                Y = Y.bmm(T)
                Z = T.bmm(Z)
            sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
            return sA

    @staticmethod
    def calculate_frechet_distances(
            mu1: torch.Tensor,
            sigma1: torch.Tensor,
            mu2: torch.Tensor,
            sigma2: torch.Tensor
    ) -> torch.Tensor:
        """
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
        Returns:
        -- dist  : The Frechet Distance.
        Raises:
        -- InvalidFIDException if nan occures.
        """
        with torch.no_grad():
            m = torch.square(mu1 - mu2).sum()
            d = torch.bmm(sigma1, sigma2)
            s = FrechetInceptionDistance.sqrtm_newton(d)
            dists = m + torch.diagonal(sigma1 + sigma2 - 2 * s, dim1=-2, dim2=-1).sum(-1)
            return dists

    def update(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> None:
        with torch.no_grad():

            mu_real, sig_real = self.calculate_activation_statistics(self.model(real_data))
            mu_fake, sig_fake = self.calculate_activation_statistics(self.model(fake_data))
            dist = self.calculate_frechet_distances(mu_real[None, :, :], sig_real[None, :, :], mu_fake[None, :, :],
                                                    sig_fake[None, :, :])
            self.distances.append(dist.item())

    def forward(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        self.update(real_data, fake_data)
        return self.distances[-1]
