
# Python libraries
import os
import numpy as np
import csv
import mne
from mne.annotations import events_from_annotations
from mne import io
from scipy.io import savemat

from data_utils.filters import bandpass_cnt


DATASET_DIRECTORY = r''
OUTPUT_DIRECTORY = r''


def save_and_annotate(
        target_dir: str,
        subject_or_session: str,
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
    subject_folder = os.path.join(target_dir, subject_or_session)
    epoch_file_path = os.path.join(subject_folder, str(epoch_idx) + '.mat')
    # # Save Sample x Channel epoch data with channel names
    savemat(epoch_file_path, {'Channels': channels, 'Samples': epoch_data})
    # Update label file
    with open(label_csv, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, label_field_names, delimiter=';')
        writer.writerow({label_field_names[0]: subject_or_session,
                         label_field_names[1]: epoch_idx,
                         label_field_names[2]: label,
                         label_field_names[3]: epoch_file_path})


def preprocess_data(source_path: str = DATASET_DIRECTORY, target_path: str = OUTPUT_DIRECTORY) -> None:
    """
    Save the already epoched set individually in csv files and create a common label file for them to be able to
    create a dataloader, thereby there won't be a need to hold the whole dataset in the RAM
    :param source_path: directory that contains the folders, in which the epched .set files are
    :param target_path: directory where the epoched csv-s will be saved
    :return: None
    """
    try:
        os.makedirs(target_path, exist_ok=True)
    except OSError as ose:
        print('Target root directory already exists!')
        exit()

    # Create and header up the label file
    label_csv_file = os.path.join(target_path, 'labels.csv')
    field_names = ['Subject', 'Epoch', 'Label', 'Path']
    with open(label_csv_file, 'w', newline='') as label_file:
        csv_writer = csv.DictWriter(label_file, fieldnames=field_names, delimiter=';')
        csv_writer.writeheader()

    # Iterate over subject dirs, load the sets and save the epochs individually
    for file_idx, file_name in enumerate(os.listdir(source_path)):
        subject_file_path = os.path.join(source_path, file_name)
        subject = int(file_name[2])

        # Only training is needed, only they are annotated
        if file_name[3] == 'E':
            continue

        # Load .gdf structure
        subject_data_struct = io.read_raw_gdf(subject_file_path, eog=[22, 23, 24], preload=True)

        # Filter data
        subject_data_struct = subject_data_struct.apply_function(
            lambda x: bandpass_cnt(x, 4., 38., 250., filt_order=3, filtfilt=False, axis=-1),
            picks=list(range(0, 22, 1))
        )

        # Extract epochs
        events, annotations = events_from_annotations(subject_data_struct)

        cue_bools = (events[:, -1] == annotations['769']) + (events[:, -1] == annotations['770']) +\
                      (events[:, -1] == annotations['771']) + (events[:, -1] == annotations['772'])
        eog_bools = (events[:, -1] == annotations['1072'])
        rej_bools = (events[:, -1] == annotations['1023'])
        exc_bools = eog_bools + rej_bools

        cue_events = [event for event in events[cue_bools] if event[0] not in events[exc_bools, 0]]

        # Extract epochs and labels
        epochs = mne.Epochs(subject_data_struct, cue_events, baseline=None, tmin=-1., tmax=4)

        epochs_data = epochs.get_data()
        labels = []
        for event in cue_events:
            if event[-1] == annotations['769']:
                labels.append('769')
            elif event[-1] == annotations['770']:
                labels.append('770')
            elif event[-1] == annotations['771']:
                labels.append('771')
            elif event[-1] == annotations['772']:
                labels.append('772')
            else:
                raise Exception("Oh-Oh")

        # Create target subject directory
        target_file_path = os.path.join(target_path, str(subject))
        try:
            os.makedirs(target_file_path)
        except OSError as ose:
            print('Target subject directory already exists!')
            exit()

        # Iterate over epochs and save them individually, also update the label file
        _ = [save_and_annotate(target_path, str(subject), label_csv_file, field_names, epoch,
                               labels[epoch_idx], np.arange(0, 25), str(epoch_idx))
             for epoch_idx, epoch in enumerate(epochs_data)]

    print('Epoched!')


if __name__ == '__main__':
    preprocess_data(DATASET_DIRECTORY, OUTPUT_DIRECTORY)
