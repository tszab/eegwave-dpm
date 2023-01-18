"""
Mock dataset to be able to initialize a Dataloader based on basic Tensor Dataset
"""

# Python imports
import torch
from torch.utils.data import Subset
from typing import List, Callable
from scipy.io import loadmat
import pandas as pd
import numpy as np
from copy import deepcopy

from ..dataset_base import DatasetBase
from ..utils import balance_data_on_class_ratios, shuffle_with_state_restore


class DummySet(DatasetBase):
    def __init__(
            self,
            set_file: str,
            transforms: List[Callable] = None,
            selected_classes: list = None,
            selected_channels: list = None,
            selected_subjects: list = None,
            selected_sessions: list = None,
            mode: str = None,
            skip_subjects: bool = True,
            name_replace: str = None,
            *args,
            **kwargs
    ):
        self._name_replace = name_replace
        super(DummySet, self).__init__(set_file, transforms, selected_classes, selected_channels, selected_subjects,
                                       selected_sessions, mode)
        self._skip_subjects = skip_subjects

    def _extract_subjects(self, loaded_data):
        """
        Extract and encode subjects
        """
        loaded_subjects = loaded_data[:, 0]
        # if self._subject_ids is None:
        #     self._subject_ids = self._get_unique(loaded_subjects)
        # subjects = torch.from_numpy(loaded_subjects.astype(np.int64))
        # if self._subject_ids[0] != 0:
        #     subjects -= 1
        # subjects = one_hot(subjects, num_classes=len(self._subject_ids)).float().numpy()
        return loaded_subjects

    def _load_paths(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        set_data = pd.read_csv(self._set_file, sep=';')
        if self._selected_sessions:
            loaded_data = set_data[['Subject', 'Path', 'Session', 'Label']].to_numpy()
        else:
            loaded_data = set_data[['Subject', 'Path', 'Label']].to_numpy()

        if self._name_replace:
            for path in range(len(loaded_data[:, 1])):
                loaded_data[path, 1] = loaded_data[path, 1].replace('diff_generated', self._name_replace)

        paths = loaded_data[:, 1]
        labels = self._extract_labels(loaded_data)
        subjects = self._extract_subjects(loaded_data)

        if self._channels is None:
            self._channels = np.array([chan.strip() for chan in loadmat(paths[0])['Channels']])
            if self._selected_channels:
                self._channels = self._channels[self._selected_channels]

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

    def split_set(
            self,
            split_ratio: float,
            split_mode: str = 'within',
            shuffle: bool = False,
            seed: int = None,
            selected_classes: list = None,
            val_split: float = None
    ) -> (Subset, Subset):
        def _shuffle_data(loaded_data: np.ndarray, seed: int) -> np.ndarray:
            set_seed = seed if seed is not None else torch.randint(0, 42, (1,)).item()
            shuffle_with_state_restore(loaded_data, set_seed, False)
            return loaded_data

        subset_indices_1 = []
        subset_indices_2 = []
        if val_split:
            assert val_split < split_ratio
            subset_indices_3 = []

        set_subjects = self._subjects

        if split_mode == 'within':
            for subject in np.unique(set_subjects):
                subject_indices = (set_subjects == subject).nonzero()[0]

                if val_split:
                    train_indices, test_indices, val_indices = \
                        balance_data_on_class_ratios(subject_indices, self._labels[subject_indices],
                                                     split_ratio, val_split, shuffle, seed)
                    subset_indices_3.extend(val_indices)
                else:
                    train_indices, test_indices = \
                        balance_data_on_class_ratios(subject_indices, self._labels[subject_indices],
                                                     split_ratio, val_split, shuffle, seed)

                subset_indices_1.extend(train_indices)
                subset_indices_2.extend(test_indices)

        elif split_mode == 'cross':
            subjects = np.unique(set_subjects)
            split_idx = int(len(subjects) * split_ratio)

            if shuffle:
                subjects = _shuffle_data(subjects, seed)

            subset_subjects_2 = subjects[split_idx:]
            if val_split:
                val_split_num = int(len(subjects) * val_split)
                subset_subjects_1 = subjects[:split_idx - val_split_num]
                subset_subjects_3 = subjects[split_idx - val_split_num:split_idx]
            else:
                subset_subjects_1 = subjects[:split_idx]

            for subject in subjects:
                if subject in subset_subjects_1:
                    subset_indices_1.extend(
                        (set_subjects == subject).nonzero()[0]
                    )
                elif subject in subset_subjects_2:
                    subset_indices_2.extend(
                        (set_subjects == subject).nonzero()[0]
                    )
                elif val_split and (subject in subset_subjects_3):
                    subset_indices_3.extend(
                        (set_subjects == subject).nonzero()[0]
                    )
        else:
            split_idx = int(self.__len__() * split_ratio)
            subset_indices_2 = [idx for idx in range(split_idx, self.__len__(), 1)]
            if val_split:
                val_split_num = int(self.__len__() * val_split)
                subset_indices_1 = [idx for idx in range(split_idx - val_split_num)]
                subset_indices_3 = [idx for idx in range(split_idx - val_split_num, split_idx, 1)]
            else:
                subset_indices_1 = [idx for idx in range(split_idx)]

        if selected_classes is not None:
            for label in selected_classes:
                subset_indices_1 = [idx for idx in (self._labels.argmax(-1) == label).nonzero()[0]
                                    if idx in subset_indices_1]
                subset_indices_2 = [idx for idx in (self._labels.argmax(-1) == label).nonzero()[0]
                                    if idx in subset_indices_2]
                if val_split:
                    subset_indices_3 = [idx for idx in (self._labels.argmax(-1) == label).nonzero()[0]
                                        if idx in subset_indices_3]

        if val_split:
            return Subset(deepcopy(self), subset_indices_1), \
                   Subset(deepcopy(self), subset_indices_2), \
                   Subset(deepcopy(self), subset_indices_3)
        else:
            return Subset(deepcopy(self), subset_indices_1), \
                   Subset(deepcopy(self), subset_indices_2)
