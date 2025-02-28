import torch
import time

from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.utils.iff_suite import IFFSuite
from fflib.nn.ff_net import FFNet
from fflib.interfaces import IFFProbe
from fflib.utils.ff_logger import logger

from typing import Dict, Any


class FFSuite(IFFSuite):
    def __init__(
        self,
        ff_net: FFNet,
        probe: IFFProbe,
        dataloader: FFDataProcessor,
        device: Any | None = None,
    ):
        super().__init__(ff_net, probe, dataloader, device)

    def _train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        y_enc = self.dataloader.encode_output(y)
        x_pos = self.dataloader.combine_to_input(x, y_enc)
        x_neg = self.dataloader.generate_negative(x, y, self.net)

        self.net.run_train(x_pos, x_neg)

    def _test(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe.predict(x)

    def train(self, epochs: int) -> None:
        logger.info("Starting Training...")
        start_time = time.time()

        for _ in range(epochs):
            super().run_train_epoch()

        # Measure the time
        end_time = time.time()
        self.time_to_train = end_time - start_time

    def test(self) -> float:
        self.test_accuracy = super().run_test_epoch(self.dataloader.get_test_loader())
        logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
        return self.test_accuracy

    def save(
        self, filepath: str, extend_dict: Dict[str, Any] = {}, append_hash: bool = False
    ) -> None:
        extend_dict.update(
            {
                "test_accuracy": self.test_accuracy,
                "time_to_train": self.time_to_train,
            }
        )

        super().save(filepath, extend_dict, append_hash)
