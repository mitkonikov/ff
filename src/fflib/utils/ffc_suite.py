import torch
import time

from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.utils.iff_suite import IFFSuite
from fflib.nn.ffc import FFC
from fflib.interfaces import IFFProbe
from fflib.utils.ff_logger import logger

from typing import Dict, Any, cast


class FFCSuite(IFFSuite):
    def __init__(
        self,
        ffc: FFC,
        probe: IFFProbe,
        dataloader: FFDataProcessor,
        device: Any | None = None,
    ):
        super().__init__(ffc, probe, dataloader, device)
        self.time_to_train: float = 0
        self.train_classifier: bool = False

    def train_switch(self, train_classifier: bool) -> None:
        self.train_classifier = train_classifier

    def _train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.train_classifier == False:
            y_enc = self.dataloader.encode_output(y)
            x_pos = self.dataloader.combine_to_input(x, y_enc)
            x_neg = self.dataloader.generate_negative(x, y, self.net)

            self.net.run_train(x_pos, x_neg)
        else:
            # Encode y through the dataloader
            y_enc = self.dataloader.encode_output(y)

            # Create the same y tensor, but all zeros
            y_flat = torch.zeros_like(y_enc).to(device=x.device)

            # Combine the x and the y_flat as an FF input
            x_pos = self.dataloader.combine_to_input(x, y_flat)

            cast(FFC, self.net).train_classifier(x_pos, y_enc)

    def _test(self, x: torch.Tensor) -> torch.Tensor:
        out_features = cast(FFC, self.net).classifier.out_features
        y_flat = torch.zeros(out_features, device=x.device)
        y_flat = y_flat.repeat(x.shape[0], 1)  # repeat it for the batch dimension
        x = self.dataloader.combine_to_input(x, y_flat)
        y: torch.Tensor = self.net(x)
        return y

    def train(self, epochs: int) -> None:
        logger.info("Starting Training...")
        start_time = time.time()

        for _ in range(epochs):
            super().run_train_epoch(validate=self.train_classifier)

        # Measure the time
        end_time = time.time()
        self.time_to_train += end_time - start_time

    def test(self) -> float:
        self.test_accuracy = super().run_test_epoch(self.dataloader.get_test_loader())
        logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
        return self.test_accuracy

    def save(
        self,
        filepath: str,
        extend_dict: Dict[str, Any] = {},
        append_hash: bool = False,
    ) -> None:
        extend_dict.update(
            {
                "test_accuracy": self.test_accuracy,
                "time_to_train": self.time_to_train,
            }
        )

        super().save(filepath, extend_dict, append_hash)
