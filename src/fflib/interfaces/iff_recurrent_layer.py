import torch

from torch.nn import Module
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List


class IFFRecurrentLayer(ABC, Module):
    @abstractmethod
    def reset_parameters(self) -> None:
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        pass

    @abstractmethod
    def goodness(
        self,
        x_prev: torch.Tensor,
        x_recc: torch.Tensor,
        x_next: torch.Tensor,
        logistic_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.log(1 + torch.exp(x)),
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Makes inference on the recurrent layer and returns the new output
        along with the goodness of the given input.

        Args:
            x_prev (torch.Tensor): Tensor representing the data coming from the prev layer at time t - 1
            x_recc (torch.Tensor): Tensor representing the data coming from the same layer at time t - 1
            x_next (torch.Tensor): Tensor representing the data coming from the next layer at time t - 1
            logistic_fn (_type_, optional): Logistic Function to decide whether the data is positive or negative.
                Defaults to lambdax:torch.log(1 + torch.exp(x)).
            inverse (bool, optional): Should we invert the output?. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First element being the goodness, second - the output of the layer
        """

        pass

    @abstractmethod
    def run_train(
        self,
        h_pos: List[torch.Tensor],
        h_neg: List[torch.Tensor],
        index: int,
    ) -> None:
        pass
