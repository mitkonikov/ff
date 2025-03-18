import torch
import os
import datetime
import time

from random import randint
from tqdm import tqdm
from torch.utils.data import DataLoader
from fflib.interfaces import IFF, IFFProbe
from fflib.utils.ff_logger import logger
from fflib.utils.data.dataprocessor import FFDataProcessor

from abc import abstractmethod
from typing import Callable, List, Dict, Tuple, Any


class IFFSuite:
    def __init__(
        self,
        ff_net: IFF,
        probe: IFFProbe,
        dataloader: FFDataProcessor,
        device: Any | None = None,
    ):
        self.net = ff_net
        self.probe = probe
        self.dataloader = dataloader

        self.device = device
        if device is not None:
            self.net.to(device)

        # Default member variables
        self.pre_epoch_callback: Callable[[IFF, int], Any] | None = None
        self.current_epoch: int = 0
        self.epoch_data: List[Dict[str, Any]] = []
        self.time_to_train: float = 0
        self.test_accuracy: float = 0

        logger.info("Created FFSuite.")

    def set_pre_epoch_callback(self, callback: Callable[[IFF, int], Any]) -> None:
        """This function allows you to hook a callback
        to be called before the training of each epoch.

        Example where this is useful is a custom LR scheduler:
        ```
        def callback(suite: FF_TestSuite, e: int):
            for i in range(0, len(suite.net.layers) - 1):
                if suite.net.layers[i] is not None:
                    cur_lr = suite.net.layers[i].get_lr()
                    next_lr = min([cur_lr, cul_lr * 2 * (1 + epochs - e) / epochs])
                    print(f"Layer {i} Next LR: {next_lr}")
                    suite.net.layers[i].set_lr(next_lr)
        ```

        Args:
            callback (Callable[[FF_TestSuite, int], Any]):
                Callback function accepting two parameters -
                The FFTestSuite object and the current epoch.
        """

        self.pre_epoch_callback = callback

    def run_test_epoch(self, loader: DataLoader[Any]) -> float:
        self.net.eval()
        test_correct: int = 0
        test_total: int = 0

        with torch.no_grad():
            for b in loader:
                batch: Tuple[torch.Tensor, torch.Tensor] = b
                x, y = batch
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)

                output = self._test(x)

                test_total += y.size(0)
                test_correct += int((output == y).sum().item())

        return test_correct / test_total

    def run_train_epoch(self, validate: bool = True) -> None:
        loaders = self.dataloader.get_all_loaders()

        # Training phase
        self.net.train()

        if self.pre_epoch_callback is not None:
            self.pre_epoch_callback(self.net, self.current_epoch)

        for b in tqdm(loaders["train"]):
            batch: Tuple[torch.Tensor, torch.Tensor] = b
            x, y = batch
            if self.device is not None:
                x, y = x.to(self.device), y.to(self.device)

            self._train(x, y)

        # Validation phase
        if validate and loaders["val"] is not None:
            val_accuracy = self.run_test_epoch(loaders["val"])
            logger.info(f"Epoch: {self.current_epoch + 1} - Val Accuracy: {val_accuracy:.4f}")
            self.epoch_data.append(
                {
                    "epoch": self.current_epoch + 1,
                    "val_accuracy": val_accuracy,
                }
            )

        self.current_epoch += 1

    @abstractmethod
    def _train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Must be implemented by derived class.
        """
        pass

    @abstractmethod
    def _test(self, x: torch.Tensor) -> torch.Tensor:
        """
        Must be implemented by derived class.
        It should accepts X and return a prediction y.
        """
        pass

    def train(self, epochs: int) -> None:
        logger.info("Starting Training...")
        start_time = time.time()

        for _ in range(epochs):
            self.run_train_epoch()

        # Measure the time
        end_time = time.time()
        self.time_to_train = end_time - start_time

    def test(self) -> float:
        self.test_accuracy = self.run_test_epoch(self.dataloader.get_test_loader())
        logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
        return self.test_accuracy

    @staticmethod
    def append_to_filename(path: str, suffix: str) -> str:
        dir_name, base_name = os.path.split(path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{name}{suffix}{ext}"
        return os.path.join(dir_name, new_filename)

    def save(
        self, filepath: str, extend_dict: Dict[str, Any] = {}, append_hash: bool = False
    ) -> str:
        data = {
            "hidden_layers": self.net.get_layer_count(),
            "current_epoch": self.current_epoch,
            "epoch_data": self.epoch_data,
            "date": str(datetime.datetime.now()),
            "net": self.net,
            "test_accuracy": self.test_accuracy,
            "time_to_train": self.time_to_train,
        }

        # Check key duplication
        for key in data.keys():
            if key in extend_dict:
                raise RuntimeError("Don't override the default FFSuite keys.")

        # Extend the data dictionary
        data.update(extend_dict)

        if append_hash:
            suffix = "_" + str(hex(randint(0, 16**6))[2:])
            filepath = self.append_to_filename(filepath, suffix)

        torch.save(data, filepath)
        return filepath

    def load(self, filepath: str) -> Any:
        """Load a pretrained FF model.

        Args:
            filepath (str): Filepath to the model.

        Returns:
            Any: IFF type of model.
        """

        data = torch.load(filepath, weights_only=False)

        for key, value in data.items():
            setattr(self, key, value)

        self.net = data["net"].to(self.device)
        return self.net
