import torch

from abc import ABC, abstractmethod


class IFFProbe(ABC):
    @abstractmethod
    def predict(
        self,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:

        pass


class NoProbe(IFFProbe):
    def predict(
        self,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:

        raise RuntimeError("Predict called on NoProbe!")
