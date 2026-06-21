import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic

import numpy as np
import pandas as pd
import torch

from components import CompSet

root = Path(__file__).resolve().parents[1]
sol_path = root / "ground_truth"

T = TypeVar('T')
S = TypeVar('S')


@dataclass
class DataPair(Generic[T, S]):
    ''' for input output pairs '''
    name: str
    input: T
    solution: S

    def __init__(self, name: str, input: T, solution: S = None):
        self.name = name
        self.input = input
        self.solution = solution


class PMPair(DataPair[torch.Tensor, torch.Tensor]):
    ''' for power map input and temp map output pairs '''

    def __init__(self, name: str, input: np.ndarray, solution: np.ndarray):
        super().__init__(
            name=name,
            input=torch.as_tensor(input, dtype=torch.float32).reshape(-1, 1),
            solution=torch.as_tensor(solution, dtype=torch.float32).reshape(-1, 1),
        )


# TODO: MAGGIE CONTEXT --> called by power_map_model
def load_pwrmp_data(load_dir=sol_path) -> list[PMPair]:
    ''' reads in all data in dir and returns list of pairs '''
    pairs = []
    p_maps = np.load(load_dir / "powermaps.npy")
    t_maps = np.load(load_dir / "temperature.npy")

    for i in range(p_maps.shape[0]):
        pairs.append(PMPair(
            name=f"Case_{i}", input=p_maps[i], solution=t_maps[i],
        ))

    return pairs


@dataclass
class ModelData(CompSet):
    pinn: list = field(default_factory=list)
    paired: list[PMPair] = field(default_factory=list)

    pinn_index: list = field(default_factory=list)
    paired_index: list = field(default_factory=list)

    def _next(self, dlist: list, index: list, pop=True):
        if len(index) == 0:
            index.extend(np.arange(len(dlist)))
            random.shuffle(index)
        if not pop:
            return dlist[index[-1]]

        return dlist[index.pop()]

    def next_pinn(self, pop=True):
        return self._next(self.pinn, self.pinn_index, pop)

    def next_pair(self, pop=True):
        return self._next(self.paired, self.paired_index, pop)

    def next(self, loss_type, pop=True):
        if loss_type == 'paired':
            return self.next_pair(pop)
        return self.next_pinn(pop)


def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def last_file(dir: Path) -> str:
    files = sorted(f for f in dir.iterdir() if f.is_file())
    return files[-1].name if files else None


def compress_dataframe(df: pd.DataFrame, x: int) -> pd.DataFrame:
    df = df.drop(columns='epoch', errors='ignore')
    chunk_ids = np.arange(len(df)) // x
    reduced = df.groupby(chunk_ids).mean().reset_index(drop=True)

    # First row index of each chunk
    reduced.insert(0, 'epoch', np.arange(len(reduced)) * x)

    return reduced
