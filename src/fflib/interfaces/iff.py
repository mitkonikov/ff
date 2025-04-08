import torch

from torch.nn import Module
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable


class IFF(ABC, Module):
    @abstractmethod
    def get_layer_count(self) -> int:
        """Return number of hidden layers in the Network

        Returns:
            int: Number of hidden layers
        """

        pass

    def _create_hooks_dict(self) -> None:
        self.hooks: Dict[str, List[Any]] = {"layer_activation": [], "layer_goodness": []}

    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        if not hasattr(self, "hooks"):
            self._create_hooks_dict()

        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
        else:
            raise ValueError(f"Hook {hook_name} not recognized.")

    def _call_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        if hook_name in self.hooks:
            for hook in self.hooks[hook_name]:
                hook(*args, **kwargs)

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
