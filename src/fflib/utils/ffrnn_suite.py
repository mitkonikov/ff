import torch

from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.utils.iff_suite import IFFSuite
from fflib.nn.ff_rnn import FFRNN
from fflib.interfaces import IFFProbe

from typing import Any


class FFRNNSuite(IFFSuite):
    def __init__(
        self,
        ffc: FFRNN,
        probe: IFFProbe,
        dataloader: FFDataProcessor,
        device: Any | None = None,
    ):
        super().__init__(ffc, probe, dataloader, device)
        self.time_to_train: float = 0

    def _train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        y_enc = self.dataloader.encode_output(y)
        x_neg, y_neg = self.dataloader.generate_negative(x, y, self.net)

        self.net.run_train(x, y_enc, x_neg, y_neg)

    def _test(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe.predict(x)
