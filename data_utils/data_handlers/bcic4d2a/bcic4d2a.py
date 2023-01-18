"""
BCI Competition 4 Dataset 2a PyTorch dataset handling class.
"""

# Project imports
import torch
from scipy.io import loadmat
import numpy as np
from torch.nn.functional import one_hot
import pandas as pd

from data_utils.dataset_base import DatasetBase


class BCIC4D2a(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(BCIC4D2a, self).__init__(*args, **kwargs)
        self._channels = np.array(['Fz',
                                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                                   'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                                   'P1', 'Pz', 'P2',
                                   'POz'])

    def _extract_epoch(self, path: str) -> torch.Tensor:
        return torch.from_numpy(loadmat(path)['Samples'])[:22]

    def _extract_labels(self, loaded_data):
        """
        Extract and encode labels
        """
        loaded_labels = loaded_data[:, -1]
        if self._classes is None:
            self._classes = self._get_unique(loaded_labels)
        labels = torch.from_numpy(loaded_labels.astype(np.int64))
        if self._classes[0] != 0:
            labels = labels - 769
        labels = one_hot(labels, num_classes=len(self._classes)).float().numpy()
        return labels

    def _extract_subjects(self, loaded_data):
        """
        Extract and encode subjects
        """
        loaded_subjects = loaded_data[:, 0]
        if self._subject_ids is None:
            self._subject_ids = np.unique(loaded_subjects)
        subjects = torch.from_numpy(loaded_subjects.astype(np.int64))
        if self._subject_ids[0] != 0:
            subjects = subjects - 1
        subjects = one_hot(subjects, num_classes=len(self._subject_ids)).float().numpy()
        return subjects

    def _load_paths(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        set_data = pd.read_csv(self._set_file, sep=';')
        if self._selected_sessions:
            loaded_data = set_data[['Subject', 'Path', 'Session', 'Label']].to_numpy()
        else:
            loaded_data = set_data[['Subject', 'Path', 'Label']].to_numpy()

        paths = loaded_data[:, 1]
        labels = self._extract_labels(loaded_data)
        subjects = self._extract_subjects(loaded_data)

        filter = np.ones(len(paths), dtype=bool)
        if self._selected_subjects is not None:
            subject_filter = self._select_subjects(loaded_data)
            filter *= subject_filter
        if self._selected_sessions is not None:
            session_filter = self._select_sessions(loaded_data)
            filter *= session_filter
        if self._selected_classes is not None:
            label_filter = self._select_classes(loaded_data)
            filter *= label_filter

        paths = paths[filter]
        labels = labels[filter]
        subjects = subjects[filter]

        return subjects, paths, labels
