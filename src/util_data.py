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
S = TypeVar('S')

@dataclass
class DataPair(Generic[T, S]):
    ''' for input output pairs '''
    name: str
    input: T
    solution: S

    def __init__(self, name: str, input: T, solution: S):
        self.name = name
        self.input = input
        self.solution = solution

    def load(self, filename, load_dir):
        return None

class PMPair(DataPair[torch.Tensor, torch.Tensor]):
    ''' for power map input and temp map output pairs '''
    # def get_tensors(self, device):
    #     return (
    #         torch.tensor(self.input.flatten(), dtype=float32).unsqueeze(-1).to(device),
    #         torch.tensor(self.solution.flatten(), dtype=float32).unsqueeze(-1).to(device)
    #     )
    def __init__(self, name: str , input: torch.Tensor , solution: torch.Tensor):
        super().__init__(
            name = name,
            input = torch.as_tensor(input, dtype=torch.float32),
            solution = torch.as_tensor(solution, dtype=torch.float32),
        )

# TODO: MAGGIE CONTEXT --> called by power_map_model
def load_pwrmp_data(load_dir=sol_path) -> list[PMPair]:
    ''' reads in all data in dir and returns list of pairs '''
    pairs = []
    p_maps = torch.as_tensor(np.load(load_dir / "powermaps.npy"), dtype=torch.float32)
    t_maps = torch.as_tensor(np.load(load_dir / "temperature.npy"), dtype=torch.float32)

    for i in range(p_maps.shape[0]):
        pairs.append(PMPair(
            name=f"Case_{i}",
            input=p_maps[i],
            solution=t_maps[i],
        ))
        
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
