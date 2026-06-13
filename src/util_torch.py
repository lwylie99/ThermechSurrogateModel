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
            n.reshape(shape)
        nps.append(n)
    return nps

def build_grid_map(dimsA:tuple, dimsB:tuple):
    ''' MAPS PLATE TO GRID coords[x,x] = [x_mm, y_mm] '''
    xs = torch.linspace(0, dimsA[0], dimsB[0])  # (20,)
    ys = torch.linspace(0, dimsA[1], dimsB[1])  # (10,)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (10, 20) — dim0=y ✓
    grid_map = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    return xs, ys, grid_map  # (200, 2)

