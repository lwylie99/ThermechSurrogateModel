import os
from dataclasses import dataclass
from typing import TypeVar, Generic, Any

import numpy as np
from torch import Tensor

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
    def load(self, filename, load_dir):
        # TODO: MAGGIE - read in from file
        self.name = filename.split('.')[0]
        self.input = None            # numpy array power map
        self.solution = np.zeros(1)   # numpy array analytical temps
        return self

# TODO: MAGGIE CONTEXT --> called by example_util
def load_pwrmp_data(load_dir) -> list[PMPair]:
    ''' reads in all data in dir and returns list of pairs '''
    pairs = []
    i = 0
    for f in os.listdir(load_dir):
        pair = DataPair().load(f, load_dir)
        pairs.append(pair)
        i += 1
    return pairs

def load_paired_data(load_dir) -> list[DataPair]:
    ''' reads in all data in dir and returns list of pairs '''
    pairs = []
    i = 0
    for f in os.listdir(load_dir):
        pair = DataPair().load(f, load_dir)
        pairs.append(pair)
        i += 1
    return pairs

def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
