import torch

from abc import ABC, abstractmethod


class IFFProbe(ABC):
    @abstractmethod
    def predict(
        self,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:

        pass
