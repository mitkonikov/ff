import torch
import os
import datetime
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randint
from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.interfaces import IFF, IFFProbe
from fflib.utils.ff_logger import logger

from typing import Callable, Any, Tuple
from typing_extensions import Self

class FFSuite:
    def __init__(
            self,
            ff_net: IFF,
            probe: IFFProbe,
            device = None
        ):
        
        self.net = ff_net
        self.probe = probe

        self.device = device
        if device is not None:
            self.net.to(device)
        
        # Default member variables
        self.pre_epoch_callback: Callable | None = None
        self.current_epoch: int = 0
        self.epoch_data: list[dict] = []

        logger.info("Created FFSuite.")

    def set_pre_epoch_callback(self, callback: Callable[[Self, int], Any]):
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

    def _validation(self, loader: DataLoader):
        if loader is not None:
            self.net.eval()
            val_accuracy: float = 0
            val_correct: int = 0
            val_total: int = 0

            with torch.no_grad():
                for b in loader:
                    batch: Tuple[torch.Tensor, torch.Tensor] = b
                    x, y = batch
                    if self.device is not None:
                        x, y = x.to(self.device), y.to(self.device)

                    output = self.probe.predict(x)

                    val_total += y.size(0)
                    val_correct += int((output == y).sum().item())

            val_accuracy = val_correct / val_total
            logger.info(f"Val Accuracy: {val_accuracy:.4f}")

            self.epoch_data.append(
                {
                    "epoch": self.current_epoch + 1,
                    "val_accuracy": val_accuracy,
                }
            )

    def train(
        self,
        dataloader: FFDataProcessor,
        epochs: int
    ):
        logger.info("Starting Training...")
        start_time = time.time()

        # Get all loaders
        loaders = dataloader.get_all_loaders()

        for _ in range(epochs):
            # Training phase
            self.net.train()

            if self.pre_epoch_callback is not None:
                self.pre_epoch_callback(self.net, self.current_epoch)

            for b in tqdm(loaders["train"]):
                batch: Tuple[torch.Tensor, torch.Tensor] = b
                x, y = batch
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)

                x_pos = dataloader.prepare_input(x, y)
                x_neg = dataloader.generate_negative(x, y, self.net)

                self.net.run_train(x_pos, x_neg)

            # Validation phase
            self._validation(loaders["val"])

            self.current_epoch += 1

        # Measure the time
        end_time = time.time()
        self.time_to_train = end_time - start_time

        return self.epoch_data

    def test(self, dataloader: FFDataProcessor):
        loader = dataloader.get_test_loader()

        test_correct: int = 0
        test_total: int = 0

        self.net.eval()
        with torch.no_grad():
            for b in loader:
                batch: Tuple[torch.Tensor, torch.Tensor] = b
                x, y = batch
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)

                output = self.probe.predict(x)

                test_total += y.size(0)
                test_correct += int((output == y).sum().item())

        test_accuracy = test_correct / test_total

        print(f"Test Accuracy: {test_accuracy:.4f}")
        self.test_data = {"test_accuracy": test_accuracy}
        return self.test_data
    
    @staticmethod
    def append_to_filename(path, suffix):
        dir_name, base_name = os.path.split(path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{name}{suffix}{ext}"
        return os.path.join(dir_name, new_filename)

    def save(self, filepath: str, append_hash: bool = False):
        data = {
            "hidden_layers": self.net.get_layer_count(),
            "current_epoch": self.current_epoch,
            "epoch_data": self.epoch_data,
            "test_data": self.test_data,
            "time_to_train": self.time_to_train,
            "date": str(datetime.datetime.now()),
            "net": self.net
        }

        if append_hash:
            suffix = "_" + str(hex(randint(0, 16**6))[2:])
            filepath = self.append_to_filename(filepath, suffix)

        torch.save(data, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)

        for key, value in data.items():
            setattr(self, key, value)

        self.net = data["net"].to(self.device)
        return self.net