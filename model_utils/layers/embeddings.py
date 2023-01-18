
# Python imports
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def continuous_timestep_embedding(timesteps, embedding_dim, max_time=1000., dtype=torch.float32):
    """Build sinusoidal embeddings (from Fairseq).
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      max_time: float: largest time input
      dtype: data type of the generated embeddings
    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    timesteps *= (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.tensor(math.log(10000.) / (half_dim - 1), device=timesteps.device, dtype=dtype)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.type(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0, 1, 0, 0])  # TODO: test this?
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class UNetStepEmbedding(nn.Module):
    def __init__(self, inp_dim: int = 128, hidden_dim: int = 512, out_dim: int = 512):
        super(UNetStepEmbedding, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.linear_1 = nn.Linear(inp_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = timestep_embedding(x, self.inp_dim).squeeze(1)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        x = F.silu(x)
        return x


class LogSNREmbedding(nn.Module):
    def __init__(self, type: str, inp_dim: int, emb_dim: int = 512, scale_range: tuple = (-10., 10.)):
        super(LogSNREmbedding, self).__init__()

        def scale_into_range(x):
            return (x - scale_range[0]) / (scale_range[1] - scale_range[0])

        def inv_cos(x):
            return torch.arctan(torch.exp(-0.5 * torch.clip(x, -20., 20.))) / (0.5 * torch.pi)

        if type == 'linear':
            self.encoder = scale_into_range
        elif type == 'sigmoid':
            self.encoder = torch.sigmoid
        elif type == 'inv_cos':
            self.encoder = inv_cos
        else:
            NotImplementedError(type)

        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.linear_1 = nn.Linear(inp_dim, emb_dim)
        self.linear_2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = continuous_timestep_embedding(x, self.inp_dim)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        x = F.silu(x)
        return x


class SinCosEmbedding(nn.Module):
    """
    Some parts of the code are taken from https://github.com/lmnt-com/diffwave.
    """
    def __init__(self, inp_dim: int = 128, hidden_dim: int = 512, out_dim: int = 512):
        super(SinCosEmbedding, self).__init__()
        self.step_embed_dim_inp = inp_dim

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        half_dim = self.step_embed_dim_inp // 2
        embed = np.log(10000) / (half_dim - 1)
        self.register_buffer('embed_part', torch.exp(torch.arange(half_dim) * -embed), persistent=False)
        self.linear_1 = nn.Linear(inp_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.embed_part
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        x = F.silu(x)
        return x


class DiffusionStepEmbedding(nn.Module):
    """
    Some parts of the code are taken from https://github.com/lmnt-com/diffwave.
    """
    def __init__(self, max_steps: int = 1000, inp_dim: int = 128, hidden_dim: int = 512, out_dim: int = 512):
        super(DiffusionStepEmbedding, self).__init__()
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.register_buffer('embedding', torch.cat([torch.sin(table), torch.cos(table)], dim=1))
        self.linear_1 = nn.Linear(inp_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def _interp_step(self, step: torch.Tensor) -> torch.Tensor:
        low_idx = step.floor().long()
        high_idx = step.ceil().long()
        low_val = self.embedding[low_idx].squeeze(1)
        high_val = self.embedding[high_idx].squeeze(1)
        return low_val + (high_val - low_val) * (step - low_idx)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        if step.dtype in [torch.long]:
            step = self.embedding[step].squeeze(1)
        elif step.dtype in [torch.float]:
            step = self._interp_step(step)
        else:
            raise TypeError('Step must be float or long Tensor!')
        step = self.linear_1(step)
        step = F.silu(step, inplace=True)
        step = self.linear_2(step)
        step = F.silu(step, inplace=True)
        return step


class DiffusionNoiseEmbedding(nn.Module):
    """
    Some parts of the code are taken from https://github.com/mindslab-ai/nuwave.
    """
    def __init__(self, scale: float = 50000, inp_dim: int = 128, hidden_dim: int = 512, out_dim: int = 512):
        super(DiffusionNoiseEmbedding, self).__init__()
        self.scale = scale

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        half_dim = inp_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / float(half_dim)
        exponents = 1e-4 ** exponents
        self.register_buffer('exponents', exponents.unsqueeze(0))

        self.linear_1 = nn.Linear(inp_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, diff_noise: torch.Tensor):
        diff_noise = diff_noise * self.scale * self.exponents
        diff_noise = torch.cat([diff_noise.sin(), diff_noise.cos()], dim=-1)
        diff_noise = self.linear_1(diff_noise)
        diff_noise = F.silu(diff_noise)
        diff_noise = self.linear_2(diff_noise)
        diff_noise = F.silu(diff_noise)
        return diff_noise
