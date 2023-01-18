
# Python imports
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Subset
from typing import List, Callable, Tuple, Optional, Union
from scipy.io import loadmat
import pandas as pd
import numpy as np
from copy import deepcopy

# Project imports
from .utils import shuffle_with_state_restore, balance_data_on_class_ratios


class DatasetBase:
    def __init__(
            self,
            set_file: str,
            transforms: List[Callable] = None,
            selected_classes: list = None,
            selected_channels: list = None,
            selected_subjects: list = None,
            selected_sessions: list = None,
            batch_mode: str = None,
            *args,
            **kwargs
    ):
        self._set_file = set_file
        self._transforms = transforms
        self._selected_classes = selected_classes
        self._selected_channels = selected_channels
        self._selected_subjects = selected_subjects
        self._selected_sessions = selected_sessions
        self._mode = batch_mode
        self._classes = None
        self._subject_ids = None
        self._channels = None
        self._subjects, self._paths, self._labels = self._load_paths()

    @property
    def channels(self):
        return self._channels

    @property
    def subjects(self):
        return self._subject_ids

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, item: int) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        epoch_data_path = self._paths[item]
        epoch_label = torch.from_numpy(self._labels[item]).float()
        if self._mode == 'subject':
            epoch_subject = torch.from_numpy(self._subjects[item]).float()

        # Load in epoch data from given csv
        epoch_data = self._extract_epoch(epoch_data_path)

        if self._selected_channels:
            epoch_data = epoch_data[self._selected_channels, :]
        epoch_data = epoch_data.unsqueeze(dim=0)

        if self._transforms is not None:
            epoch_data = epoch_data.double()
            for transform in self._transforms:
                epoch_data = transform(epoch_data)
                if not isinstance(epoch_data, torch.Tensor):
                    epoch_data = torch.from_numpy(epoch_data).double()
            epoch_data = epoch_data.float()

        if self._mode == 'subject':
            return epoch_data.float(), epoch_label, epoch_subject
        else:
            return epoch_data.float(), epoch_label

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

        set_subjects = self._subjects.argmax(-1)

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

    def _extract_epoch(self, path: str) -> torch.Tensor:
        return torch.from_numpy(loadmat(path)['Samples'])

    def _select_subjects(self, loaded_data: np.ndarray) -> np.ndarray:
        selected_subject_indices = (loaded_data[:, 0] == self._selected_subjects[0])
        if len(self._selected_subjects) > 1:
            for subject in self._selected_subjects[1:]:
                selected_subject_indices += (loaded_data[:, 0] == subject)
        return np.array(selected_subject_indices)

    def _select_sessions(self, loaded_data: np.ndarray) -> np.ndarray:
        selected_session_indices = (loaded_data[:, -2] == self._selected_classes[0])
        if len(self._selected_sessions) > 1:
            for session_id in self._selected_sessions[1:]:
                selected_session_indices += (loaded_data[:, -2] == session_id)
        return np.array(selected_session_indices)

    def _select_classes(self, loaded_data: np.ndarray) -> np.ndarray:
        selected_class_indices = (loaded_data[:, -1] == self._selected_classes[0])
        if len(self._selected_classes) > 1:
            for label in self._selected_classes[1:]:
                selected_class_indices += (loaded_data[:, -1] == label)
        return np.array(selected_class_indices)

    def _get_unique(self, data):
        data = np.unique(data)
        if len(data) < 2:
            data = np.array([0, 1])
            print('DATASET CONTAINS DATA ONLY FROM ONE CLASS!')
        return data

    def _extract_subjects(self, loaded_data):
        """
        Extract and encode subjects
        """
        loaded_subjects = loaded_data[:, 0]
        if self._subject_ids is None:
            self._subject_ids = self._get_unique(loaded_subjects)
        subjects = torch.from_numpy(loaded_subjects.astype(np.int64))
        subjects = one_hot(subjects, num_classes=len(self._subject_ids)).float().numpy()
        return subjects

    def _extract_labels(self, loaded_data):
        """
        Extract and encode labels
        """
        loaded_labels = loaded_data[:, -1]
        if self._classes is None:
            self._classes = self._get_unique(loaded_labels)
        labels = torch.from_numpy(loaded_labels.astype(np.int64))
        labels = one_hot(labels, num_classes=len(self._classes)).float().numpy()
        return labels

    def _load_paths(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        set_data = pd.read_csv(self._set_file, sep=';')
        if self._selected_sessions:
            loaded_data = set_data[['Subject', 'Path', 'Session', 'Label']].to_numpy()
        else:
            loaded_data = set_data[['Subject', 'Path', 'Label']].to_numpy()

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
