"""
Model configs
"""

# Python imports
from functools import partial

# Project import
from models import *
from evaluation.baselines.VAEs.models.ozan_vae import OzanVAE
from evaluation.baselines.GANs.models.rwgan import RWGAN


MODELS = {
    'eegwave': partial(EEGWave, eeg_channels=44, inp_channels=1, out_channels=44,
                       res_channels=64, skip_channels=64, res_layers=32, dilation_cycles=11, kernel_size=3),
    'unet': partial(UNetModel, inp_channels=1, out_channels=8, image_size=64,
                    model_channels=8, num_res_blocks=2, attention_resolutions="16,8",
                    channel_mult=(1, 2, 3, 4), use_scale_shift_norm=True),
    'ozanvae': partial(OzanVAE, inp_channels=1, out_channels=64, hidden_dim=64, seq_length=64),
    'rwgan': partial(RWGAN, gen_inp_dim=120, gen_out_dim=1, gen_hid_dim=64, disc_inp_dim=1, disc_out_dim=1,
                     disc_hid_dim=64, shape=(44, 256))
}
