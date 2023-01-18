"""
Main file for general training and evaluation of the used diffusion models. Parameters are given in run_config.py.
"""


# Python imports
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
from ptflops import get_model_complexity_info

# Project imports
from configurations import *
from run_config import *
from torch.utils.data import DataLoader
from data_utils.utils import time_plot, spectral_plot


if __name__ == '__main__':
    try:
        def init_dataloader(dataset) -> DataLoader:
            return DataLoader(dataset, DATASET_CFG['batch_size'],
                              DATASET_CFG['shuffle'], num_workers=HW_PARAMS['num_workers'],
                              prefetch_factor=HW_PARAMS['pre_fetch'])

        # Initialize dataset and dataloaders
        dc = DATASET_CFG
        dataset = DATASETS[DATASET_CFG['dataset']](**dc)
        if (dc['val_ratio'] is not None) and (dc['val_ratio'] > 0.0):
            train_set, eval_set, _ = dataset.split_set(dc['train_test_ratio'], dc['split_mode'], dc['shuffle'],
                                                       dc['shuffle_seed'], None,
                                                       val_split=dc['val_ratio'])
        else:
            train_set, eval_set = dataset.split_set(dc['train_test_ratio'], dc['split_mode'], dc['shuffle'],
                                                    dc['shuffle_seed'], None,
                                                    val_split=dc['val_ratio'])

        train_loader = init_dataloader(train_set)
        eval_loader = init_dataloader(eval_set)

        subjects = np.unique([train_set.dataset.subjects, eval_set.dataset.subjects])

        fc = FRAMEWORK_CFG

        # Initialize model, optimizer, loss and scheduler
        checkpoint = RUN_CFG['checkpoint_path']
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location='cpu')

        start_epoch = 0
        model_name = fc['model']
        optimizer_name = fc['optimizer']
        loss_name = fc['loss']
        scheduler_name = fc['scheduler']

        model = MODELS[model_name](classes=fc['classes'], subjects=len(subjects) if fc['subject_embed'] else None)
        if HW_PARAMS['device'] != 'cpu':
            model.cuda()

        if RUN_CFG['mode'] == 'info':
            # from torchsummary import summary
            # summary(model, (1, 1, 240), device='cpu')

            macs, params = get_model_complexity_info(model.cpu(), RUN_CFG['generate_shape'], as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            print('MACS: ' + str(macs))
            print('PARAMS: ' + str(params))
            exit()

        if checkpoint is not None:
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']

        optimizer = OPTIMIZERS[optimizer_name](params=model.parameters())
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = SCHEDULERS[scheduler_name](optimizer=optimizer) if scheduler_name is not None else None
        if checkpoint is not None:
            if checkpoint['scheduler'] is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])

        checkpoint = None

        # Initialize framework
        fc['model'] = model
        fc['optimizer'] = optimizer
        fc['scheduler'] = scheduler
        framework = fc['framework'](**fc, subjects=len(subjects) if fc['subject_embed'] else None)

        if RUN_CFG['ema_checkpoint_path'] is not None:
            framework.set_target_state(
                torch.load(RUN_CFG['ema_checkpoint_path'], map_location='cpu')['model']
            )

        # Run selected program mode
        if RUN_CFG['mode'] == 'train':
            # Initialize logger
            logger = LOGGER_CFG['logger'](log_dir=LOGGER_CFG['log_dir'], save_only_target=LOGGER_CFG['save_only_target'],
                                          save_model_epoch_freq=LOGGER_CFG['save_model_epoch_freq'],
                                          sig_fs=LOGGER_CFG['sig_fs'], fid_model=LOGGER_CFG['fid_model'],
                                          is_model=LOGGER_CFG['is_model'])

            # Subscribe to training events with the logger
            framework.publisher.on_batch_end.append(lambda x: logger.log_batch_end(x))
            framework.publisher.on_epoch_train_end.append(lambda x: logger.log_epoch_train_end(x))
            framework.publisher.on_epoch_val_end.append(lambda x: logger.log_epoch_val_end(x))
            framework.publisher.on_epoch_val_end.append(lambda _: logger.reset_batch_eval_list())

            # Start training process
            framework.train(RUN_CFG["epochs"], train_loader, start_epoch=start_epoch, eval_freq=RUN_CFG['eval_freq'],
                            eval_samples_num=RUN_CFG['generate_samples'], micro_batch=DATASET_CFG['micro_batch'])

        elif RUN_CFG['mode'] == 'generate' or RUN_CFG['mode'] == 'compare_sets':
            # Generate fake samples
            data_dir = os.path.join(os.path.split(dataset._set_file)[0], RUN_CFG['generated_dir'])
            os.makedirs(data_dir)

            print('Generating fake samples')
            fake_samples, labels, subjects = framework.generate(RUN_CFG['generate_samples'], RUN_CFG['generate_shape'],
                                                                DATASET_CFG['batch_size'], FRAMEWORK_CFG['ddim_infer'],
                                                                RUN_CFG['generate_classes'], RUN_CFG['generate_subjects'])

            fake_samples = fake_samples.squeeze(1).numpy()
            if labels is not None:
                labels = labels.argmax(-1).numpy()
            if subjects is not None:
                subjects = subjects.argmax(-1).numpy() + 1

            # Save generated data into the original directory
            label_csv = os.path.join(data_dir, 'generated_labels.csv')
            field_names = ['Subject', 'Epoch', 'Label', 'Path']

            subject_list = np.unique(subjects) if subjects is not None else ['96']
            with open(label_csv, 'w', newline='') as label_file:
                csv_writer = csv.DictWriter(label_file, fieldnames=field_names, delimiter=';')
                csv_writer.writeheader()
                for subject in subject_list:
                    epoch_dir = os.path.join(data_dir, str(subject))
                    os.makedirs(epoch_dir)
                    if len(subject_list) > 1:
                        epochs_to_save = fake_samples[subjects == subject]
                        labels_to_save = labels[subjects == subject] if labels is not None else None
                    else:
                        epochs_to_save = fake_samples
                        labels_to_save = labels if labels is not None else None
                    for epoch_idx, epoch_data in enumerate(epochs_to_save):
                        epoch_file_path = os.path.join(epoch_dir, str(epoch_idx) + '.mat')
                        # # Save Sample x Channel epoch data with channel names
                        savemat(epoch_file_path, {'Channels': list(dataset.channels), 'Samples': epoch_data})
                        # Update label file
                        csv_writer.writerow({field_names[0]: str(subject),
                                             field_names[1]: epoch_idx,
                                             field_names[2]: labels_to_save[epoch_idx]
                                                                if labels_to_save is not None else (RUN_CFG['generate_classes'][0]
                                                                if RUN_CFG['generate_classes'] is not None else 0),
                                             field_names[3]: epoch_file_path})

            if RUN_CFG['mode'] == 'compare_sets':
                print('Loading real samples')
                real_samples = []
                with tqdm(total=len(train_loader), leave=True) as pbar:
                    for batch in train_loader:
                        real_data = batch[0]
                        real_samples.append(real_data)
                        pbar.update(1)
                real_samples = torch.cat(real_samples, dim=0).squeeze(1).numpy()

                time_plot(real_samples, fake_samples)
                plt.savefig(os.path.join(data_dir, 'time.png'))

                plt.cla()
                plt.clf()

                spectral_plot(real_samples, fake_samples, LOGGER_CFG['sig_fs'])
                plt.savefig(os.path.join(data_dir, 'spectra.png'))

                plt.cla()
                plt.clf()

                chan_list = list(dataset.channels)
                from data_utils.utils import compare_scalp
                compare_scalp([real_samples, fake_samples], ['Real signals', 'Synthesized signals'],
                              chan_list,
                              LOGGER_CFG['sig_fs'] - 1, np.linspace(0., 1., 10),
                              path=os.path.join(data_dir, 'topomap_over_time'),
                              v_limits=LOGGER_CFG['top_plot_limits'])
                compare_scalp([real_samples, fake_samples], ['Time-Averaged Real signals', 'Time-Averaged Synthesized signals'],
                              chan_list,
                              LOGGER_CFG['sig_fs'] - 1, 1.0, average=1.0,
                              path=os.path.join(data_dir, 'topomap_avg_time'),
                              v_limits=LOGGER_CFG['avg_plot_limits'])
    except RuntimeWarning:
        pass
