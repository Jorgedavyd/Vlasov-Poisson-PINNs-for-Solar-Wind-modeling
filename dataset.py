## define all related to the dataset
from modulus.sym.domain.constraint.discrete import DictGridDataset
from modulus.sym.hydra import ModulusConfig
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import pandas as pd
from torch import Tensor
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset


def define_data(
    real_data: NDArray, real_time: NDArray, cfg: ModulusConfig
) -> Tuple[Tensor, Tensor]:
    time_grid = np.arange(0, 1, cfg.custom.geometry.grid_resolution)
    interpolator = CubicSpline(real_time, real_data)
    dataset = torch.from_numpy(interpolator(time_grid))
    return torch.from_numpy(time_grid), dataset
