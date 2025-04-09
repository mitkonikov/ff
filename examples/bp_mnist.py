# %% Imports
import torch
import os
import sys
import argparse

from torch.optim import Adam, Optimizer

# Get the absolute path of the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fflib.interfaces.iff import IFF
from fflib.utils.data.mnist import FFMNIST
from fflib.utils.bp_suite import BPSuite
from fflib.utils.ff_logger import logger

from typing import cast

# It should skip the argument parser if it finds it runs within an IPython shell.
parser = argparse.ArgumentParser(exit_on_error=False)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=30, type=int)
parser.add_argument("-u", "--use", help="Part of MNIST to use", default=1.0, type=float)
args = parser.parse_args(args=("" if "get_ipython" in globals() else None))


class BPDenseNet(IFF):
    def __init__(self, lr: float):
        super().__init__()

        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 10),
        ]

        self.lr = lr
        self.layers = torch.nn.Sequential(*layers)
        self.criterion = torch.nn.CrossEntropyLoss()

        self._init_utils()

    def _init_utils(self) -> None:
        self.opt: Optimizer | None = Adam(self.parameters(), self.lr)

    def get_layer_count(self) -> int:
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(x))

    def run_train_combined(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> None:
        raise NotImplementedError("Use run_train function with separate X and Y data inputs.")

    def run_train(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> None:

        if not hasattr(self, "opt") or self.opt == None:
            raise ValueError("Optimizer is not set!")

        self.opt.zero_grad()
        output = self.forward(x_pos)
        loss = self.criterion(output, y_pos)
        loss.backward()
        self.opt.step()

    def strip_down(self) -> None:
        self.opt = None


# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

torch.manual_seed(42)

# %% Setup Dataset
logger.info("Setting up MNIST dataset...")
mnist = FFMNIST(batch_size=256, validation_split=0.1, use=args.use)

# %% Setup Dense Backpropagation Network
net = BPDenseNet(0.001)
suite = BPSuite(net, mnist, device)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
suite.test()

# %% Save Model
logger.info("Saving model...")
net.strip_down()
suite.save("./models/bp_mnist.pt", append_hash=True)
