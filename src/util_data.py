import os
from dataclasses import dataclass
from typing import TypeVar, Generic, Any

import numpy as np
import torch
from torch import Tensor, float32

from pathlib import Path

root = Path(__file__).resolve().parents[1]
sol_path = root / "ground_truth"

T = TypeVar('T')

@dataclass
class DataPair(Generic[T]):
    ''' for input output pairs '''
    name: str = None
    input: T = None
    solution: np.ndarray = None

    def load(self, filename, load_dir):
        return None

class PMPair(DataPair[np.ndarray]):
    ''' for power map input and temp map output pairs '''
    def get_tensors(self, device):
        return (
            torch.tensor(self.input.flatten(), dtype=float32).unsqueeze(-1).to(device),
            torch.tensor(self.solution.flatten(), dtype=float32).unsqueeze(-1).to(device)
        )

# TODO: MAGGIE CONTEXT --> called by power_map_model
def load_pwrmp_data(load_dir=sol_path) -> list[PMPair]:
    ''' reads in all data in dir and returns list of pairs '''
    pairs = []
    p_maps = np.load(load_dir / "powermaps.npy")
    t_maps = np.load(load_dir / "temperature.npy")
    case_names = [f"Case_{i}" for i in range(p_maps.shape[0])]
    
    for i in range(p_maps.shape[0]):
        pair = PMPair(
            name=case_names[i],
            input=p_maps[i],
            solution=t_maps[i],
        )
        pairs.append(pair)
        
    return pairs

# Merged load_pwrmp and load_paired so commenting this for now
# def load_paired_data(load_dir) -> list[DataPair]:
#     ''' reads in all data in dir and returns list of pairs '''
#     pairs = []
#     i = 0
#     for f in os.listdir(load_dir):
#         pair = DataPair().load(f, load_dir)
#         pairs.append(pair)
#         i += 1
#     return pairs

def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
