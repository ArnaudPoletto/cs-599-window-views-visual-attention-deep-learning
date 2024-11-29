import random
import torch
import numpy as np
import lightning.pytorch as pl


def set_seed(seed: int) -> None:
    """
    Set the random seed for all relevant packages.

    Args:
        seed (int): The seed to use
    """
    pl.seed_everything(seed, workers=True)
