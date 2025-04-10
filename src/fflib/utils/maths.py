import torch

from fflib.enums import SparsityType

from math import sqrt, log2


def ComputeSparsity(x: torch.Tensor, type: SparsityType) -> torch.Tensor:
    n = x.shape[0]
    if type == SparsityType.HOYER:
        r = torch.sum(torch.abs(x)) / torch.sqrt(torch.sum(torch.square(x)))
        sqn = sqrt(n)
        return torch.Tensor((sqn - r) / (sqn - 1))
    elif type == SparsityType.ENTROPY_BASED:
        t = torch.sum(torch.abs(x))
        p = torch.abs(x).div(t) + 1e-8
        v = p * torch.log2(p)
        h = -torch.sum(v)
        return torch.Tensor(1 - (h / log2(n)))
