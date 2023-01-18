import mne
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, List
from mne.baseline import rescale
from mne.filter import resample


def shuffle_with_state_restore(data: np.ndarray, seed: int = None, copy: bool = False) -> np.ndarray:
    if copy:
        shuffled_data = data.copy()
    else:
        shuffled_data = data

    # Save current state
    current_rnd_state = np.random.get_state()
    if seed:
        np.random.seed(seed)
    np.random.shuffle(shuffled_data)

    # Restore state
    np.random.set_state(current_rnd_state)
    return shuffled_data


def std_norm_data(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    std, mean = torch.std_mean(data, dim=dim, keepdim=True)
    return (data - mean) / std


def min_max_norm_data(data: torch.Tensor, dim: int = -1, min_max: tuple = (-1., 1.)) -> torch.Tensor:
    nominator = (data - data.min(dim=dim, keepdim=True)[0]) * (min_max[1] - min_max[0])
    divisor = data.max(dim=dim, keepdim=True)[0] - data.min(dim=dim, keepdim=True)[0]
    return min_max[0] + nominator / divisor


def mean_norm_data(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (data - data.mean(dim=dim, keepdim=True)[0]) / \
           (data.max(dim=dim, keepdim=True)[0] - data.min(dim=dim, keepdim=True)[0])


def labeled_tube_plot(x, data_y, tube_y, labels,
                      title="", xlabel="", ylabel="", axes=None):
    x = np.asarray(x)
    if axes is None:
        axes = plt.gca()

    colors = []
    for i, label in enumerate(labels):
        y_tmp = data_y[i]
        tube_tmp = tube_y[i]
        p = axes.fill_between(x, y_tmp + tube_tmp, y_tmp - tube_tmp, alpha=0.5, label=labels[i])
        # p = axes.fill_between(x, y_tmp, y_tmp, alpha=0.5, label=labels[i])
        colors.append(p._original_facecolor)

    for i, label in enumerate(labels):
        axes.plot(x, data_y[i], lw=2, color=colors[i])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(x.min(), x.max())

    data_y_cat = np.concatenate(data_y, axis=0)
    tube_y_cat = np.concatenate(tube_y, axis=0)
    axes.set_ylim((data_y_cat - tube_y_cat).min(), (data_y_cat + tube_y_cat).max())
    axes.legend()


def compute_spectral_amplitude(x, axis=None):
    fft = np.fft.rfft(x, axis=axis)
    return np.log(np.abs(fft) + 1e-6)


def calc_rms(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.pow(data, 2), dim=dim))


def spectral_plot(X_real: np.ndarray, X_fake: np.ndarray, fs, axes=None):
    n_samples = X_real.shape[2]
    freqs = np.fft.rfftfreq(n_samples, 1. / fs)
    amps_real = compute_spectral_amplitude(X_real, axis=2)
    amps_real_mean = amps_real.mean(axis=(0, 1)).squeeze()
    amps_real_std = amps_real.std(axis=(0, 1)).squeeze()
    amps_fake = compute_spectral_amplitude(X_fake, axis=2)
    amps_fake_mean = amps_fake.mean(axis=(0, 1)).squeeze()
    amps_fake_std = amps_fake.std(axis=(0, 1)).squeeze()
    labeled_tube_plot(freqs,
                      [amps_real_mean, amps_fake_mean],
                      [amps_real_std, amps_fake_std],
                      ["Real", "Fake"],
                      "Mean spectral log amplitude", "Hz", "log(Amp)", axes)


def time_plot(X_real: np.ndarray, X_fake: np.ndarray, axes=None):
    times = np.arange(X_real.shape[-1])
    amps_real_mean = X_real.mean(axis=(0, 1)).squeeze()
    amps_real_std = X_real.std(axis=(0, 1)).squeeze()
    amps_fake_mean = X_fake.mean(axis=(0, 1)).squeeze()
    amps_fake_std = X_fake.std(axis=(0, 1)).squeeze()
    labeled_tube_plot(times,
                      [amps_real_mean, amps_fake_mean],
                      [amps_real_std, amps_fake_std],
                      ["Real", "Fake"],
                      "Mean amplitude", "s", "Amp", axes)


def extract_epoch(signal: torch.Tensor, time_limits: Tuple[float, float], signal_length: float):
    sample_num = signal.shape[-1]
    start_idx = int(sample_num / signal_length * time_limits[0])
    stop_idx = int(sample_num / signal_length * time_limits[1])
    return signal[..., start_idx:stop_idx]


def scalp_plot(
        dataset: np.ndarray,
        ch_names: list,
        sfreq: float,
        times: Union[float, np.ndarray],
        average: float = None,
        std_1020: bool = True,
        title: str = None,
        path: str = None,
        v_limits: tuple = (-20., 20.),
        res: int = 600
) -> None:
    ch_type = 'eeg'
    eeg_info = mne.create_info(ch_names, sfreq, ch_type, verbose=False)
    try:
        if std_1020:
            eeg_info.set_montage('standard_1020')
    except:
        eeg_info.set_montage('standard_1005')
    evoked = mne.EvokedArray(dataset.mean(0) / 10e+3, eeg_info)
    evoked.plot_topomap(ch_type=ch_type, times=times, average=average, colorbar=True,
                        time_unit='s', res=res, title=title, vmax=v_limits[1], vmin=v_limits[0])
    if path is not None:
        plt.savefig(path)


def compare_scalp(
        datasets: List[np.ndarray],
        dataset_names: List[str],
        ch_names: List[str],
        sfreq: float,
        times: Union[float, np.ndarray],
        average: float = None,
        std_1020: bool = True,
        path: str = None,
        v_limits: tuple = (-20., 20.),
        res: int = 600
) -> None:
    plot_num = len(datasets)

    for plot_idx in range(plot_num):
        scalp_plot(datasets[plot_idx], ch_names, sfreq, times, average, std_1020, dataset_names[plot_idx],
                   path + dataset_names[plot_idx], v_limits, res)


def extract_seq(x: torch.Tensor, limits: Tuple[int, int]) -> torch.Tensor:
    return x[..., limits[0]:limits[1]]


def resample_tensor(
        data: torch.Tensor,
        down: float = 1.,
        up: float = 1.,
        npad: int = 'auto',
        dim: int = -1
) -> torch.Tensor:
    return torch.from_numpy(resample(data.numpy(), down=down, up=up, npad=npad, axis=dim))


def baseline_tensor(
        data: torch.Tensor,
        times: list,
        baseline: tuple,
        mode: str = 'mean',
) -> torch.Tensor:
    return torch.from_numpy(rescale(data.numpy(), times=times, baseline=baseline, mode=mode, verbose=False))


def balance_data_on_class_ratios(
        data: np.ndarray,
        labels: np.ndarray,
        train_test_split: float,
        val_split: float,
        shuffle: bool,
        shuffle_seed: int
) -> (np.ndarray, np.ndarray):
    labels_int = labels.argmax(-1)
    class_num = max(len(np.unique(labels_int)), labels.shape[-1])
    classes = list(range(class_num))
    class_elements = [np.sum(labels_int == label) for label in classes]
    class_indices = [(labels_int == label).nonzero()[0] for label in classes]

    if shuffle:
        for class_type in class_indices:
            shuffle_with_state_restore(class_type, seed=shuffle_seed)

    train_split_indices = [int(train_test_split * class_el) for class_el in class_elements]

    test_indices = np.concatenate(
        [type_indices[train_split_indices[class_type_idx]:]
         for class_type_idx, type_indices in enumerate(class_indices)],
        axis=-1
    )
    if val_split:
        val_split_indices = [train_split_indices[class_idx] - int(val_split * class_el)
                             for class_idx, class_el in enumerate(class_elements)]
        val_indices = np.concatenate(
            [type_indices[val_split_indices[class_type_idx]:train_split_indices[class_type_idx]]
             for class_type_idx, type_indices in enumerate(class_indices)],
            axis=-1
        )
        train_indices = np.concatenate(
            [type_indices[:val_split_indices[class_type_idx]]
             for class_type_idx, type_indices in enumerate(class_indices)],
            axis=-1
        )
    else:
        train_indices = np.concatenate(
            [type_indices[:train_split_indices[class_type_idx]]
             for class_type_idx, type_indices in enumerate(class_indices)],
            axis=-1
        )

    if val_split:
        return data[train_indices], data[test_indices], data[val_indices]
    else:
        return data[train_indices], data[test_indices]
