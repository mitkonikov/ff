import torch

from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, ToTensor, Lambda  # type: ignore

from fflib.utils.data import FFDataProcessor
from fflib.interfaces.iff import IFF

from enum import Enum
from typing import Tuple, Dict, Callable, Any


class NegativeGenerator(Enum):
    INVERSE = 1
    RANDOM = 2
    HIGHEST_INCORRECT = 3


class FFMNIST(FFDataProcessor):
    def __init__(
        self,
        batch_size: int,
        validation_split: float | None,
        download: bool = True,
        path: str = "./data",
        image_transform: Callable[..., Any] = Compose([ToTensor(), Lambda(torch.flatten)]),
        train_kwargs: Dict[str, Any] = {},
        test_kwargs: Dict[str, Any] = {},
        negative_generator: NegativeGenerator = NegativeGenerator.INVERSE,
        use: float = 1.0,
    ):

        assert isinstance(batch_size, int)
        assert batch_size > 0
        self.batch_size = batch_size
        if "batch_size" not in train_kwargs:
            train_kwargs["batch_size"] = self.batch_size
        if "batch_size" not in test_kwargs:
            test_kwargs["batch_size"] = self.batch_size

        train_kwargs["shuffle"] = True

        assert use >= 0.0 and use <= 1.0

        self.validation_split = validation_split
        self.download = download
        self.path = path
        self.image_transform = image_transform
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
        self.negative_generator = negative_generator
        self.use = use

        self.train_dataset = MNIST(
            self.path, train=True, download=self.download, transform=self.image_transform
        )
        self.test_dataset = MNIST(
            self.path, train=False, download=self.download, transform=self.image_transform
        )
        self.test_loader = DataLoader(self.test_dataset, **self.test_kwargs)

        dataset_size = len(self.train_dataset)
        used_dataset_size = int(dataset_size * self.use)
        not_used_dataset_size = dataset_size - used_dataset_size

        # In case a validation split is given
        if self.validation_split:
            # Determine the sizes of training and validation sets
            val_size = int(self.validation_split * used_dataset_size)
            train_size = used_dataset_size - val_size

            # Split dataset into train and validation sets
            train_dataset, val_dataset, _ = random_split(
                self.train_dataset, [train_size, val_size, not_used_dataset_size]
            )

            # Create data loaders for train and validation
            self.train_loader = DataLoader(train_dataset, **self.train_kwargs)
            self.val_loader = DataLoader(val_dataset, **self.test_kwargs)

            assert len(self.train_loader) + len(self.val_loader) <= used_dataset_size
            return

        train_dataset, _ = random_split(
            self.train_dataset, [used_dataset_size, not_used_dataset_size]
        )
        self.train_loader = DataLoader(train_dataset, **self.train_kwargs)

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
        return one_hot(y, num_classes=10).float()

    def combine_to_input(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y), 1)

    def generate_negative(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        net: IFF,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.negative_generator == NegativeGenerator.HIGHEST_INCORRECT:
            raise NotImplementedError()

        if self.negative_generator == NegativeGenerator.INVERSE:
            y_hot = 1 - one_hot(y, num_classes=10).float()
            return x, y_hot

        rnd = torch.rand((x.shape[0], 10), device=x.device)
        rnd[torch.arange(x.shape[0]), y] = 0
        y_new = rnd.argmax(1)
        y_hot = one_hot(y_new, num_classes=10).float()
        return x, y_hot
