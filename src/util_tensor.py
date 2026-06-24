import numpy as np
import torch
from torch import Tensor


def init_torch_device(device=None):
    if device is None:
        device = "cuda"
    device_str = device if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    return torch.device(device_str)

def to_numpy(tns:list[Tensor], shape:tuple=None):
    nps = []
    for t in tns:
        n = t.cpu().detach().numpy()
        if shape is not None:
            n = n.reshape(shape)
        nps.append(n)
    return nps

def build_grid_map(dimsA:tuple, dimsB:tuple, offset:tuple=(0.0,0.0)):
    ''' MAPS PLATE TO GRID coords[x,x] = [x_mm, y_mm] '''
    xs = torch.linspace(0, dimsA[0], dimsB[0]) + offset[0]
    ys = torch.linspace(0, dimsA[1], dimsB[1]) + offset[1]
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid_map = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    return xs, ys, grid_map

def normalize(tensor, vmin=None, vmax=None):
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    print(f'normalize: target=({vmin}, {vmax}), tensor range=[{t_min}, {t_max}]')
    if t_max == t_min:
        return torch.zeros_like(tensor)
    # Scale to [0, 1] first, then scale into [vmin, vmax]
    normalized = (tensor - t_min) / (t_max - t_min)
    if vmin is not None and vmax is not None:
        normalized = normalized * (vmax - vmin) + vmin
    return normalized

def normalize_np(arr, vmin=None, vmax=None):
    nmin, nmax = np.min(arr), np.max(arr)
    normal = (arr - nmin) / (vmax - nmax)
    if vmin is not None and vmax is not None:
        normal = normal * (vmax - vmin) + vmin
    return normal