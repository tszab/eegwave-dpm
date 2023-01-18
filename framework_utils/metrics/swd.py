
import torch
from torch import nn
import numpy as np


class SlicedWassersteinDistance(nn.Module):
    """
    Originally from https://github.com/kahartma/eeggan to get comparable results.
    """
    def __init__(self, n_projections: int, n_features: int):
        super(SlicedWassersteinDistance, self).__init__()
        self.n_projections = n_projections
        self.n_features = n_features
        self.w_transform = self._create_wasserstein_transform_matrix(n_projections, n_features)
        self.distances = []

    @staticmethod
    def _create_wasserstein_transform_matrix(n_projections, n_features) -> torch.Tensor:
        return np.random.normal(size=(n_projections, n_features))

    def _calculate_sliced_wasserstein_distance(self, real_data: torch.Tensor, fake_date: torch.Tensor) -> torch.Tensor:
        input1 = real_data
        input2 = fake_date

        if input1.shape[0] != input2.shape[0]:
            n_inputs = input1.shape[0] if input1.shape[0] < input2.shape[0] else input2.shape[0]
            input1 = np.random.permutation(input1)[:n_inputs]
            input2 = np.random.permutation(input2)[:n_inputs]

        input1 = input1.reshape(input1.shape[0], -1)
        input2 = input2.reshape(input2.shape[0], -1)

        transformed1 = np.matmul(self.w_transform, input1.T)
        transformed2 = np.matmul(self.w_transform, input2.T)

        for i in np.arange(self.w_transform.shape[0]):
            transformed1[i] = np.sort(transformed1[i], -1)
            transformed2[i] = np.sort(transformed2[i], -1)

        diff = transformed1 - transformed2
        diff = np.power(diff, 2).mean()
        return np.sqrt(diff)

    def reset(self) -> None:
        self.w_transform = self._create_wasserstein_transform_matrix(self.n_projections, self.n_features)

    def forward(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        self.reset()
        distance = self._calculate_sliced_wasserstein_distance(
            real_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else real_data,
            fake_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else fake_data
        )
        self.distances.append(distance)
        return distance
