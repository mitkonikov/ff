import torch

from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.utils.iff_suite import IFFSuite
from fflib.nn.ff_net import FFNet
from fflib.interfaces import IFFProbe

from typing import Any


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

        self.net.run_train_combined(x_pos, x_neg)

    def _test(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe.predict(x)
