"""
Main file for generating interpolated data based on the combination of individual classes.
"""

# Python imports
import os
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.functional import one_hot
from tqdm import tqdm
from functools import partial
import csv
from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.utils.data import DataLoader

# Project imports
from configurations import *
from run_config import *
from configurations.models import MODELS
from data_utils.utils import scalp_plot


# Main parameters
OUTPUT_PTH = os.path.join(os.getcwd(), RUN_CFG['generated_dir'])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COMBINE_NUM = 13  # Combination of N classes
INTERP_ALPHAS = [1.0]  # Interpolation ratio


def main():
    # Gather parameters
    dc = DATASET_CFG
    set_name = dc['dataset']
    shape = RUN_CFG['generate_shape']
    selected_classes = RUN_CFG['generate_classes']
    class_num = FRAMEWORK_CFG['classes']
    selected_subjects = RUN_CFG['generate_subjects']
    batch_size = dc['batch_size']
    plot_original = False
    plot_interpolated = False

    # Channels and subject num can be given
    channels = None
    subject_num = None

    # Go through the combination of classes and generate data along the combined conditions
    interp_subjects = []
    interp_names = []
    for combination in itertools.combinations(selected_subjects, COMBINE_NUM):
        comb_loader_dict = {}
        comb_label_list = []

        # Collect the data from the individual classes in a given combination
        for comb_item in combination:
            dataset_a = DATASETS[set_name](transforms=dc['transforms'],
                                           selected_channels=dc['selected_channels'],
                                           selected_classes=selected_classes,
                                           selected_subjects=[comb_item+1])

            dataloader_a = DataLoader(dataset_a, dc['batch_size'])
            comb_loader_dict.update({str(comb_item+1): dataloader_a})

            if channels is None:
                channels = list(dataset_a.channels)
            if subject_num is None:
                subject_num = len(dataset_a.subjects)

            # Save combination labels
            a = torch.tensor(comb_item, device=DEVICE).long()
            a = one_hot(a, num_classes=subject_num).float()
            comb_label_list.append(a)

        # Create output folders and plots of the original data from the individual classes
        comb_labels_sum = torch.stack(comb_label_list, dim=0).sum(dim=0)
        for comb_label in comb_label_list:
            other_comb_sum = comb_labels_sum - comb_label

            for alpha in INTERP_ALPHAS:
                slerped = slerp(comb_label, other_comb_sum, alpha, COMBINE_NUM - 1)
                interp_subjects.append(slerped)
                interp_names.append(
                    f'{comb_label.argmax(-1).item() + 1}_' + \
                    '_'.join([str(elem.item() + 1) for elem in (other_comb_sum > 0).nonzero()]) + \
                    f'_{alpha}'
                )

                subject_dir = os.path.join(OUTPUT_PTH, interp_names[-1])
                os.makedirs(subject_dir)

                if plot_original:
                    plot_compare_data(
                        comb_loader_dict,
                        subject_dir,
                        channels
                    )

    # Initialize model
    model = MODELS[FRAMEWORK_CFG['model']](classes=class_num, subjects=subject_num, interpolate=True)
    model.load_state_dict(torch.load(RUN_CFG['checkpoint_path'], map_location='cpu')['model'])
    model.cuda() if DEVICE == 'cuda' else model.cpu()
    diffusion = FRAMEWORK_CFG['diffusion']
    sampler = FRAMEWORK_CFG['sampler']

    # Generate interpolated data based on the combination of the classes
    generated_samples, generated_labels, generated_subjects, interp_names \
        = generate_interpolate(model, diffusion, sampler, RUN_CFG['generate_samples'], shape, batch_size, True,
                               selected_classes, interp_subjects, interp_names)

    # Save generated data into the original directory
    save_data(generated_samples, generated_labels, interp_names, channels, plot_interpolated)


def slerp(z1: torch.Tensor, z2: torch.Tensor, alpha: float, z2_div: float = 1.) -> torch.Tensor:
    """
    Combine the one-hot encoded labels
    """
    return alpha * z1 + ((1 - alpha) / z2_div) * z2


def generate_interpolate(
        model: nn.Module,
        diffusion: object,
        sampler: object,
        samples: int,
        shape: Tuple,
        batch_size: int = 1,
        ddim_infer: bool = True,
        classes: list = None,
        subjects: list = None,
        interp_names: list = None
) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    """
    Function for handling the generation process of the interpolated data
    """
    def _set_device(to_set: Union[nn.Module, torch.Tensor]) -> Union[nn.Module, torch.Tensor]:
        to_set = to_set.cuda() if DEVICE != 'cpu' else to_set.cpu()
        return to_set

    model.eval()

    infer_type = 'ddim' if ddim_infer else 'noisy'

    # Define the number of iterations
    samples = samples * ((len(subjects) if subjects else 1) * (len(classes) if classes else 1))
    batch_steps = (samples + (batch_size - (samples % batch_size))) // batch_size

    if classes:
        # Generate batch_step number of samples from all classes
        labels = torch.tensor(classes * (samples // len(classes)))
        labels = one_hot(labels, num_classes=FRAMEWORK_CFG['classes']).float()
        labels = _set_device(labels)
        generated_labels = []

    if subjects:
        # Generate batch_step number of samples from all classes
        subj = torch.stack(subjects).repeat(samples // len(subjects), 1)
        subj = _set_device(subj)
        generated_subjects = []

        interp_names = interp_names * (samples // len(subjects))

    generated_samples = []
    with tqdm(total=batch_steps, leave=True) as pbar:
        for batch_step in range(batch_steps):
            curr_idx = batch_step * batch_size
            batch_labels = labels[curr_idx:curr_idx + batch_size] if classes else None
            batch_subjects = subj[curr_idx:curr_idx + batch_size] if subjects else None

            if classes:
                gen_size = batch_labels.shape[0]
            elif subjects:
                gen_size = batch_subjects.shape[0]
            else:
                gen_size = batch_size
            data = torch.randn(size=(gen_size, *shape), device=DEVICE)
            data = diffusion.sample_loop(
                partial(model, label=batch_labels, subject=batch_subjects),
                data, sampler, infer_type, False
            )

            data = torch.clamp(data, -1., 1.)

            generated_samples.append(data)
            if classes:
                generated_labels.append(batch_labels)
            if subjects:
                generated_subjects.append(batch_subjects)
            pbar.update(1)

    generated_samples = torch.cat(generated_samples, dim=0).cpu()
    generated_labels = torch.cat(generated_labels, dim=0).cpu() if classes else None
    generated_subjects = torch.cat(generated_subjects, dim=0).cpu() if subjects else None

    # Return synthetic data in one big batch
    return generated_samples, generated_labels, generated_subjects, interp_names


def save_data(
        generated_samples: torch.Tensor,
        generated_labels: torch.Tensor,
        interp_names: list,
        channels: list,
        plot: bool
) -> None:
    # Save generated data into the original directory
    label_csv = os.path.join(OUTPUT_PTH, 'generated_labels.csv')
    field_names = ['Subject', 'Epoch', 'Label', 'Path']

    with open(label_csv, 'w', newline='') as label_file:
        csv_writer = csv.DictWriter(label_file, fieldnames=field_names, delimiter=';')
        csv_writer.writeheader()

        sort_indices = np.argsort(interp_names)
        generated_samples = generated_samples[sort_indices].squeeze(1).numpy()
        generated_labels = generated_labels[sort_indices].argmax(-1).numpy()

        for idx, subject_name in enumerate(np.array(interp_names)[sort_indices]):
            subject_dir = os.path.join(OUTPUT_PTH, subject_name)
            os.makedirs(subject_dir, exist_ok=True)

            if 0 < idx < (len(interp_names) - 1):
                if prev_subj_name != subject_name:
                    if plot:
                        # Create scalp plot
                        plot_generated_data(generated_samples[idx - epoch_cnt:idx],
                                            os.path.join(OUTPUT_PTH, prev_subj_name), channels)

                    epoch_cnt = 0
                    prev_subj_name = subject_name
                else:
                    epoch_cnt += 1
            elif idx == (len(interp_names) - 1) and plot:
                # Create scalp plot
                plot_generated_data(generated_samples[idx - epoch_cnt:idx], subject_dir, channels)
            else:
                epoch_cnt = 0
                prev_subj_name = subject_name

            epoch_file_path = os.path.join(subject_dir, 'data')
            os.makedirs(epoch_file_path, exist_ok=True)
            epoch_file_path = os.path.join(epoch_file_path, str(epoch_cnt) + '.mat')
            # # Save Sample x Channel epoch data with channel names
            savemat(epoch_file_path, {'Channels': channels, 'Samples': generated_samples[idx]})
            # Update label file
            csv_writer.writerow({field_names[0]: subject_name,
                                 field_names[1]: epoch_cnt,
                                 field_names[2]: generated_labels[idx].item()
                                 if generated_labels is not None
                                 else (RUN_CFG['generate_classes'][0]
                                       if RUN_CFG['generate_classes'] is not None
                                       else 0),
                                 field_names[3]: epoch_file_path})


def plot_generated_data(samples: np.ndarray, subject_dir: str, channels: list):
    plt.cla()
    plt.clf()
    plt.close()
    scalp_plot(dataset=samples,
               ch_names=channels,
               sfreq=LOGGER_CFG['sig_fs'],
               times=np.linspace(0., 1., 10),
               title='Interpolated',
               path=os.path.join(subject_dir, 'top_plot_set_interpolated.png'),
               v_limits=LOGGER_CFG['top_plot_limits'])
    plt.cla()
    plt.clf()
    plt.close()
    scalp_plot(dataset=samples,
               ch_names=channels,
               sfreq=LOGGER_CFG['sig_fs'],
               average=1.0,
               times=1.0,
               title='Set_{}'.format('Generated'),
               path=os.path.join(subject_dir, 'top_plot_avg_set_generated.png'),
               v_limits=LOGGER_CFG['avg_plot_limits'])


def plot_compare_data(datasets: dict, dir_pth: str, channels: list):
    sfreq = LOGGER_CFG['sig_fs']

    for d_name, d_value in datasets.items():
        real_samples = []
        for batch in d_value:
            real_samples.append(batch[0])
        real_samples = torch.cat(real_samples, dim=0).squeeze(1).numpy()

        plt.cla()
        plt.clf()
        plt.close()

        scalp_plot(dataset=real_samples, ch_names=channels, sfreq=sfreq,
                   times=np.linspace(0., 1., 10), title='Set_{}'.format(d_name),
                   path=os.path.join(dir_pth, 'top_plot_set_{}.png'.format(d_name)),
                   v_limits=LOGGER_CFG['top_plot_limits'])

        plt.cla()
        plt.clf()
        plt.close()

        scalp_plot(dataset=real_samples, ch_names=channels, sfreq=sfreq,
                   average=1.0, times=1.0, title='Set_{}'.format(d_name),
                   path=os.path.join(dir_pth, 'top_plot_avg_set_{}.png'.format(d_name)),
                   v_limits=LOGGER_CFG['avg_plot_limits'])

        plt.cla()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()












