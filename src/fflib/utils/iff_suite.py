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


class IFFSuite:
    def __init__(self, ff_net: IFF, probe: IFFProbe, device=None):
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

    @staticmethod
    def append_to_filename(path, suffix):
        dir_name, base_name = os.path.split(path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{name}{suffix}{ext}"
        return os.path.join(dir_name, new_filename)

    def _save(self, filepath: str, extend_dict: dict, append_hash: bool = False):
        data = {
            "hidden_layers": self.net.get_layer_count(),
            "current_epoch": self.current_epoch,
            "epoch_data": self.epoch_data,
            "date": str(datetime.datetime.now()),
            "net": self.net,
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

    def _load(self, filepath: str):
        data = torch.load(filepath)

        for key, value in data.items():
            setattr(self, key, value)

        self.net = data["net"].to(self.device)
        return self.net
