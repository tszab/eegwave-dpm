"""
Originally from https://github.com/kahartma/eeggan
"""

# Python imports
import torch
from torch import nn
import numpy as np


class InceptionScore(nn.Module):
    def __init__(self, model: nn.Module, splits: int = 1, repetitions: int = 100):
        super(InceptionScore, self).__init__()
        self.model = model
        self.splits = splits
        self.repetitions = repetitions
        self.stats = []

    def reset(self) -> None:
        self.stats = []

    @staticmethod
    def calculate_inception_score(preds, splits, repetitions) -> (float, float):
        with torch.no_grad():
            stepsize = np.max((int(np.ceil(preds.size(0) / splits)), 2))
            steps = np.arange(0, preds.size(0), stepsize)
            scores = []
            for rep in np.arange(repetitions):
                preds_tmp = preds[torch.randperm(preds.size(0), device=preds.device)]
                if len(preds_tmp) < 2:
                    continue
                for i in np.arange(len(steps)):
                    preds_step = preds_tmp[steps[i]:steps[i] + stepsize]
                    step_mean = torch.mean(preds_step, 0, keepdim=True)
                    kl = preds_step * (torch.log(preds_step) - torch.log(step_mean))
                    kl = torch.mean(torch.sum(kl, 1))
                    scores.append(torch.exp(kl).item())
                # Calculating the inception score
                # part = preds[preds.argmax(-1) == 0].view(-1, 1).cpu().numpy()
                # logp = np.log(part)
                # self = np.sum(part * logp, axis=1)
                # cross = np.mean(np.dot(part, np.transpose(logp)), axis=1)
                # diff = self - cross
                # kl = np.mean(self - cross)
                # kl1 = []
                # for j in range(splits):
                #     diffj = diff[(j * diff.shape[0] // splits):((j + 1) * diff.shape[0] // splits)]
                #     kl1.append(np.exp(diffj.mean()))
                # scores.append(np.exp(kl))
            return np.mean(scores).item(), np.std(scores).item()

    def update(self, fake_data: torch.Tensor) -> None:
        with torch.no_grad():
            preds = self.model(fake_data)
            # preds[preds.argmax(-1) == 1, 1] = 0.99
            # preds[preds.argmax(-1) == 1, 0] = 0.01
            # preds[preds.argmax(-1) != 1, 1] = 0.01
            # preds[preds.argmax(-1) != 1, 0] = 0.99
            # preds = torch.exp(preds)
            score_mean, score_std = self.calculate_inception_score(preds, self.splits, self.repetitions)
        self.stats.append([score_mean, score_std])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update(x)
        return self.stats[-1]

