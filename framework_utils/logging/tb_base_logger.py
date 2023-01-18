
# Python libraries
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Project imports
from data_utils.utils import spectral_plot, time_plot


class SignalTBLogger:
    """
    Class that responsible for data logging into the tensorboard
    """
    def __init__(self, log_dir: str, fs: float):
        log_dir = os.path.join(log_dir, 'sig_tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.fs = fs

    def log_loss(self, step: int, loss: float, extra_tag: str = None) -> None:
        tag = 'Loss/'
        if extra_tag:
            tag += extra_tag
        self.writer.add_scalar(tag, loss, step)

    def log_pred(
            self,
            step: int,
            inputs: np.ndarray,
            generations: np.ndarray,
            real_labels: np.ndarray = None,
            real_subjects: np.ndarray = None,
            fake_labels: np.ndarray = None,
            fake_subjects: np.ndarray = None,
            extra_tag: str = None
    ) -> None:
        use_labels = (real_labels is not None and fake_labels is not None)

        plt.cla()
        plt.close('all')

        if use_labels:
            classes = np.unique(np.concatenate([real_labels, fake_labels], axis=0), axis=0)
            for label in classes:
                reals_to_plot = inputs[label.argmax(-1) == real_labels.argmax(-1)]
                fakes_to_plot = generations[label.argmax(-1) == fake_labels.argmax(-1)]

                if len(reals_to_plot.shape) > 3:
                    reals_to_plot = reals_to_plot.squeeze(0)
                    fakes_to_plot = fakes_to_plot.squeeze(0)
                elif len(reals_to_plot.shape) < 3:
                    reals_to_plot = np.expand_dims(reals_to_plot, 0)
                    fakes_to_plot = np.expand_dims(fakes_to_plot, 0)

                try:
                    fig = plt.figure()
                    ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])
                    time_plot(reals_to_plot, fakes_to_plot)
                    ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
                    spectral_plot(reals_to_plot, fakes_to_plot, self.fs)

                    tag = 'Sample plots for class' + str(label.argmax(-1))
                    tag = tag if extra_tag is None else tag + '/' + extra_tag
                    self.writer.add_figure(tag, fig, step)
                except Exception as e:
                    print('Plot could not be generated...')
        else:
            reals_to_plot = inputs
            fakes_to_plot = generations

            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])
            time_plot(reals_to_plot, fakes_to_plot)
            ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
            spectral_plot(reals_to_plot, fakes_to_plot, self.fs)

            tag = 'Sample plots'
            tag = tag if extra_tag is None else tag + '/' + extra_tag
            self.writer.add_figure(tag, fig, step)

    def log_h_params(self, h_params: dict, metrics: dict) -> None:
        self.writer.add_hparams(h_params, metrics)

    def log_model_graph(
            self,
            model: torch.nn.Module,
            input: torch.Tensor = None,
    ) -> None:
        self.writer.add_graph(model, input, verbose=False)

    def log_generative_metrics(self, step: int, generative_scores: dict):
        for k, v in generative_scores.items():
            self.writer.add_scalar('Metrics/' + k, v, step)
