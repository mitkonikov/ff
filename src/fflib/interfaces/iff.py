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
        self.hooks: Dict[str, Dict[str, Callable[..., Any]]] = {
            "layer_activation": {},
            "layer_goodness": {},
        }

    def register_hook(self, hook_tag: str, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a custom hook inside the network.

        Args:
            hook_tag (str): Predefined hook tag that corresponds to some event in the network
            hook_name (str): Custom user-defined hook name, so the user can override the callback without having to save IDs
            callback (Callable[..., Any]): Callback to be called on some hook_tag event

        Raises:
            ValueError: Raises error if the hook_tag is not defined.
        """
        if not hasattr(self, "hooks") or len(self.hooks.keys()) == 0:
            self._create_hooks_dict()

        if hook_tag in self.hooks:
            self.hooks[hook_tag][hook_name] = callback
        else:
            raise ValueError(f"Hook Tag {hook_tag} not recognized.")

    def _call_hooks(self, hook_tag: str, *args: Any, **kwargs: Any) -> None:
        if hook_tag in self.hooks:
            for hook in self.hooks[hook_tag].keys():
                self.hooks[hook_tag][hook](*args, **kwargs)

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
