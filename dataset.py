## define all related to the dataset
import numpy as np
from numpy.typing import NDArray
from typing import List
import pandas as pd
from torch import Tensor
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

def get_data_grid(real_data: NDArray, real_time: NDArray, time_grid: NDArray) -> Tensor:
    interpolator = CubicSpline(real_time, real_data)
    dataset = torch.from_numpy(interpolator(time_grid))
    return dataset

def get_time_grid(resolution: float) -> NDArray:
    return np.arange(0, 1, resolution)

class L1Dataset(Dataset):
    def __init__(self, path: str, output_var: List[str]) -> None:
        df: pd.DataFrame = pd.read_csv(path).loc[output_var, :]
        df["t"] = np.array(list(map(lambda x: x.seconds, df.index - df.index[0])))
        data: NDArray = df.values
        real_data: NDArray = data
        real_time: NDArray = data["t"]
        self.keys = output_var
        self.interpolator = CubicSpline(real_time, real_data)

    def __getitem__(self, t):
        values: NDArray = self.interpolator(t)
        return {
            key: values[idx] for idx, key in enumerate(self.keys)
        }
