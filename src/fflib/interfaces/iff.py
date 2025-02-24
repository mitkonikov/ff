import torch

from torch.nn import Module
from abc import ABC, abstractmethod

class IFF(ABC, Module):
    @abstractmethod
    def get_layer_count(self) -> int:
        pass

    @abstractmethod
    def run_train(
            self,
            x_pos: torch.Tensor,
            x_neg: torch.Tensor,
        ):

        pass
