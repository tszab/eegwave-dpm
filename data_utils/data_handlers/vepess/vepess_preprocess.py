"""
This file contains the implementation of the epoch extraction of BCIC3D2a data set.
"""

# Python libraries
import os
import numpy as np
import csv
from scipy.io import savemat
import mne


DATASET_DIRECTORY = r''
OUTPUT_DIRECTORY = r''

F_SAMPLING = 512
CHANNEL_NUM = 64


def save_and_annotate(
        target_dir: str,
        subject: str,
        label_csv: str,
        label_field_names: list,
        epoch_data: np.ndarray,
        label: str,
        channels: np.ndarray,
        epoch_idx: str
) -> None:
    """
    A quick implementation for epoch file and label file saving
    """
    subject_folder = os.path.join(target_dir, subject)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
    epoch_file_path = os.path.join(subject_folder, str(epoch_idx) + '.mat')
    # # Save Sample x Channel epoch data with channel names
    savemat(epoch_file_path, {'Channels': channels, 'Samples': epoch_data})
    # Update label file
    with open(label_csv, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, label_field_names, delimiter=';')
        writer.writerow({label_field_names[0]: subject,
                         label_field_names[1]: epoch_idx,
                         label_field_names[2]: label,
                         label_field_names[3]: epoch_file_path})


def preprocess_data(source_path: str, target_path: str):
    """
    Epochs the recorded dataset into individual csv-s and session folders, also creates a main label file that contains
    the information for every epoch and their paths.
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Create and header up the label file
    label_csv_file = os.path.join(target_path, 'labels.csv')
    field_names = ['Subject', 'Epoch', 'Label', 'Path']
    with open(label_csv_file, 'w', newline='') as label_file:
        csv_writer = csv.DictWriter(label_file, fieldnames=field_names, delimiter=';')
        csv_writer.writeheader()

    # Iterate over subject dirs, load the sets and save the epochs individually
    for subject_file in os.listdir(source_path):
        subject_file_path = os.path.join(source_path, subject_file)
        subject_num = os.path.splitext(subject_file)[0].split('_')[1]

        raw = mne.io.read_epochs_eeglab(subject_file_path)
        raw_data = raw.get_data()

        target_event_ids = []
        nontarget_event_ids = []
        for key, item in raw.event_id.items():
            if '34' in key:
                nontarget_event_ids.append(item)
            elif '35' in key:
                target_event_ids.append(item)

        nontarget_event_indices = []
        target_event_indices = []
        for item in nontarget_event_ids:
            nontarget_event_indices.extend((item == raw.events[:, -1]).nonzero()[0])
        for item in target_event_ids:
            target_event_indices.extend((item == raw.events[:, -1]).nonzero()[0])

        target_data = raw_data[target_event_indices]
        nontarget_data = raw_data[nontarget_event_indices]
        subject_data = np.concatenate((target_data, nontarget_data), axis=0)

        target_labels = np.ones(len(target_data))
        nontarget_labels = np.zeros(len(nontarget_data))
        subject_labels = np.concatenate((target_labels, nontarget_labels), axis=0)

        # Iterate over epochs and save them individually, also update the label file
        _ = [save_and_annotate(target_path, str(subject_num), label_csv_file, field_names, epoch,
                               str(subject_labels[epoch_idx]), raw.info['ch_names'], str(epoch_idx))
             for epoch_idx, epoch in enumerate(subject_data)]

    print('Epoched!')


if __name__ == '__main__':
    preprocess_data(DATASET_DIRECTORY, OUTPUT_DIRECTORY)
