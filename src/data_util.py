import os
from dataclasses import dataclass
from typing import TypeVar, Generic

from torch import Tensor

T = TypeVar('T')


@dataclass
class DataPair(Generic[T]):
    ''' for input output pairs '''
    name: str = None
    input: T = None
    solution: Tensor = None

    def load_solution(self, filename, load_dir):
        self.name = filename.split('.')[0]
        # TODO: read in from file (MAGGIE)
        self.solution = None
        return self


def load_paired_data(input_data: list, load_dir):
    pairs = []
    i = 0
    for f in os.listdir(load_dir):
        pair = DataPair(input_data[0]).load_solution(f, load_dir)
        pairs.append(pair)
        i += 1

    return pairs

def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
