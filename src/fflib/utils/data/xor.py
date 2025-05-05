import torch

from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader

from fflib.utils.data import FFDataProcessor
from fflib.interfaces.iff import IFF

from typing import Tuple, Dict, Any


class XORDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = torch.randint(0, 2, (2,)).long()
        label = int(sample[0].item()) ^ int(sample[1].item())
        return sample, label


class FFXOR(FFDataProcessor):
    def __init__(
        self,
        batch_size: int,
        size: int = 100,
        train_kwargs: Dict[str, Any] = {},
        test_kwargs: Dict[str, Any] = {},
    ):

        assert isinstance(batch_size, int)
        assert batch_size > 0
        self.batch_size = batch_size
        if "batch_size" not in train_kwargs:
            train_kwargs["batch_size"] = self.batch_size
        if "batch_size" not in test_kwargs:
            test_kwargs["batch_size"] = self.batch_size

        train_kwargs["shuffle"] = True

        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs

        self.train_loader = DataLoader(XORDataset(size), **self.train_kwargs)
        self.test_loader = DataLoader(XORDataset(size), **self.test_kwargs)
        self.val_loader = DataLoader(XORDataset(size), **self.test_kwargs)

    def get_input_shape(self) -> torch.Size:
        return torch.Size((2,))

    def get_output_shape(self) -> torch.Size:
        return torch.Size((2,))

    def get_train_loader(self) -> DataLoader[Any]:
        return self.train_loader

    def get_val_loader(self) -> DataLoader[Any]:
        return self.val_loader

    def get_test_loader(self) -> DataLoader[Any]:
        return self.test_loader

    def get_all_loaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            "train": self.get_train_loader(),
            "val": self.get_val_loader(),
            "test": self.get_test_loader(),
        }

    def encode_output(self, y: torch.Tensor) -> torch.Tensor:
        return one_hot(y, num_classes=2).float()

    def combine_to_input(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y), 1)

    def generate_negative(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        net: IFF,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        rnd = torch.rand((x.shape[0], 2), device=x.device)
        rnd[torch.arange(x.shape[0]), y] = 0
        y_new = rnd.argmax(1)
        y_hot = one_hot(y_new, num_classes=2).float()
        return x, y_hot
