import torch

from fflib.utils.data.dataprocessor import FFDataProcessor
from fflib.utils.iff_suite import IFFSuite
from fflib.interfaces.iff import IFF
from fflib.interfaces.iffprobe import NoProbe
from fflib.probes.identity import Identity

from typing import Any


class BPSuite(IFFSuite):
    def __init__(
        self,
        bp_net: IFF | str,
        dataloader: FFDataProcessor,
        device: Any | None = None,
    ):
        super().__init__(bp_net, NoProbe(), dataloader, device)
        self.probe = Identity(lambda x: torch.argmax(self.net(x), 1))

    def _train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        y_enc = self.dataloader.encode_output(y)
        self.net.run_train(x, y_enc, torch.Tensor(), torch.Tensor())

    def _test(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe.predict(x)
