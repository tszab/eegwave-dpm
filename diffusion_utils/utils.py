"""
This code is taken from https://github.com/openai/guided-diffusion and partly modified.
"""
import numpy as np
import math
from functools import partial
import torch as th


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_limits=(0.0001, 0.02)):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * beta_limits[0]
        beta_end = scale * beta_limits[1]
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_logsnr_schedule(name, **kwargs):
    """Get log SNR schedule (t==0 => logsnr_max, t==1 => logsnr_min)."""
    schedules = {
        'cosine': _logsnr_schedule_cosine,
        'linear': _logsnr_schedule_linear
    }
    return partial(schedules[name], **kwargs)


def _betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
    b = th.arctan(th.exp(th.tensor(-0.5 * logsnr_max, device=t.device, dtype=t.dtype)))
    a = th.arctan(th.exp(th.tensor(-0.5 * logsnr_min, device=t.device, dtype=t.dtype))) - b
    return -2. * th.log(th.tan(a * t + b))


def _logsnr_schedule_linear(t, *, logsnr_min, logsnr_max):
    return t * (logsnr_max - logsnr_min) + logsnr_min
