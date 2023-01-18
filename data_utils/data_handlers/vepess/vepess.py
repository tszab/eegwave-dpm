"""
VEPESS PyTorch dataset handling class.
"""

import torch
from torch.nn.functional import one_hot
from scipy.io import loadmat
import numpy as np

# Project imports
from data_utils.dataset_base import DatasetBase


class VEPESS(DatasetBase):
    def __init__(self, *args, **kwargs):
        self._channels = None  # need to be set
        super(VEPESS, self).__init__(*args, **kwargs)

    def _extract_epoch(self, path: str) -> torch.Tensor:
        return torch.from_numpy(loadmat(path)['Samples'])

    def _extract_subjects(self, loaded_data):
        """
        Extract and encode subjects
        """
        loaded_subjects = loaded_data[:, 0]
        if self._subject_ids is None:
            self._subject_ids = self._get_unique(loaded_subjects)
        subjects = torch.from_numpy(loaded_subjects.astype(np.int64))
        if self._subject_ids[0] != 0:
            subjects -= 1
        subjects = one_hot(subjects, num_classes=len(self._subject_ids)).float().numpy()
        return subjects
