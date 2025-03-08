import torch

from torch.nn import Module
from abc import ABC, abstractmethod


class IFF(ABC, Module):
    @abstractmethod
    def get_layer_count(self) -> int:
        """Return number of hidden layers in the Network

        Returns:
            int: Number of hidden layers
        """

        pass

    @abstractmethod
    def run_train_combined(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> None:

        pass

    @abstractmethod
    def run_train(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> None:

        pass
