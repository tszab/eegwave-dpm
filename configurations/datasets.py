"""
Dataset path dictionary
"""

from functools import partial
from data_utils import data_handlers as dl


DATASETS = {
    'bcic4d2a':
        partial(dl.BCIC4D2a,
                set_file=r'...\labels.csv'),  # need to be set
    'vepess':
        partial(dl.VEPESS,
                set_file=r'...\labels.csv'),  # need to be set
}
