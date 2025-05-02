import torch

from fflib.enums import SparsityType

from math import sqrt, log2
from typing import Dict, cast


def ComputeSparsity(x: torch.Tensor, type: SparsityType) -> torch.Tensor:
    x = torch.flatten(x)
    n = x.shape[0]
    if type == SparsityType.HOYER:
        r = x.norm(1) / x.norm(2)
        sqn = sqrt(n)
        return torch.Tensor((sqn - r) / (sqn - 1))
    elif type == SparsityType.L1_NEG_ENTROPY:
        len = cast(torch.Tensor, x.norm(1))
        p = x.abs() / len
        v = p * p.log2()
        return v.sum() / log2(n) + 1
    elif type == SparsityType.L2_NEG_ENTROPY:
        len = cast(torch.Tensor, x.norm(2).square())
        p = x.square() / len
        v = p * p.log2()
        return v.sum() / log2(n * n) + 1
    elif type == SparsityType.GINI:
        x = x.abs()
        len = cast(torch.Tensor, x.norm(1))
        x_sorted = x.sort().values
        idx = torch.arange(1, n + 1, 1, device=x.device)
        p = (x_sorted / len) * (n - idx + 0.5)
        return -p.sum() * 2 / n + 1


def ComputeAllSparsityTypes(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    result = {}
    for type in SparsityType:
        result[str(type).split(".")[1]] = ComputeSparsity(x, type)
    return result
