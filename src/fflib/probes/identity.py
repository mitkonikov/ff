import torch

from fflib.interfaces.iffprobe import IFFProbe

from typing import Callable


class Identity(IFFProbe):
    def __init__(self, callback: Callable[[torch.Tensor], torch.Tensor]):
        self.callback = callback

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.callback(x)
