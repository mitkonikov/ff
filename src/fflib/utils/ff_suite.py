import torch
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
from fflib.utils.iff_suite import IFFSuite
from fflib.nn.ff_net import FFNet
from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.interfaces import IFFProbe
from fflib.utils.ff_logger import logger

from typing import Tuple, List, Dict, Any


class FFSuite(IFFSuite):
    def __init__(self, ff_net: FFNet, probe: IFFProbe, device: Any | None = None):
        super().__init__(ff_net, probe, device)

    def _validation(self, loader: DataLoader[Any]) -> None:
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

    def train(self, dataloader: FFDataProcessor, epochs: int) -> List[Dict[str, Any]]:
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

    def test(self, dataloader: FFDataProcessor) -> Dict[str, Any]:
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

    def save(self, filepath: str, append_hash: bool = False) -> None:
        data = {
            "test_data": self.test_data,
            "time_to_train": self.time_to_train,
        }

        super()._save(filepath, data, append_hash)

    def load(self, filepath: str) -> Any:
        """Load a pretrained FF model.

        Args:
            filepath (str): Filepath to the model.

        Returns:
            Any: IFF type of model.
        """
        return super()._load(filepath)
