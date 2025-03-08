import torch

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from fflib.interfaces.iff import IFF

from typing import Tuple, Dict, Any


class FFDataProcessor(ABC):
    @staticmethod
    def check_splits(splits: Tuple[float, float, float] | Tuple[float, float]) -> None:
        assert abs(sum(splits) - 1) < 1e-4
        assert all(0 <= s <= 1 for s in splits)
        assert len(splits) in [2, 3]

    @abstractmethod
    def get_train_loader(self) -> DataLoader[Any]:
        pass

    @abstractmethod
    def get_val_loader(self) -> DataLoader[Any] | None:
        pass

    @abstractmethod
    def get_test_loader(self) -> DataLoader[Any]:
        pass

    @abstractmethod
    def get_all_loaders(self) -> Dict[str, DataLoader[Any]] | Dict[str, Any]:
        """This function should return a dictionary containing 2 or 3 dataloader
        in the following form:

        ```
        {
            "train": train_loader,
            "val": val_loader | None,
            "test": test_loader
        }
        ```

        All of the loaders should be of type `torch.utils.data.DataLoader`.

        Returns:
            Dict[str, DataLoader] | Dict[str, Any]: Dictionary containing loaders.
        """
        pass

    @abstractmethod
    def encode_output(self, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def combine_to_input(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_negative(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        net: IFF,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
