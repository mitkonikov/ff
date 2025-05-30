# %% Imports
import torch

import sys
import os
import argparse

# Get the absolute path of the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fflib.nn.ff_linear import FFLinear
from fflib.nn.ff_net import FFNet
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.data.mnist import FFMNIST, NegativeGenerator as MNISTNEG
from fflib.utils.ff_suite import FFSuite
from fflib.utils.ff_logger import logger

# It should skip the argument parser if it finds it runs within an IPython shell.
parser = argparse.ArgumentParser(exit_on_error=False)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=60, type=int)
parser.add_argument("-u", "--use", help="Part of MNIST to use", default=1.0, type=float)
args = parser.parse_args(args=("" if "get_ipython" in globals() else None))

# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

torch.manual_seed(42)

# %% Setup Dataset
logger.info("Setting up MNIST dataset...")
mnist = FFMNIST(
    batch_size=128,
    validation_split=0.1,
    negative_generator=MNISTNEG.RANDOM,
    use=args.use,
)

# %% Setup the layers
logger.info("Setting up layers...")
lt = 20
lr = 0.02

layer1 = FFLinear(
    in_features=10 + 28 * 28,
    out_features=2000,
    loss_threshold=lt,
    lr=lr,
    device=device,
)

layer2 = FFLinear(
    in_features=2000,
    out_features=2000,
    loss_threshold=lt,
    lr=lr,
    device=device,
)

# Setup a basic network
logger.info("Setting up FFNet...")
net = FFNet([layer1, layer2], device)

# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: net(torch.cat((x, y), 1)), output_classes=10)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFSuite(net, probe, mnist, device)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
suite.test()

# %% Save Model
logger.info("Saving model...")
model_path = suite.save("./models/ff_net_mnist.pt", append_hash=True)

# %% Load Model
for i in range(len(net.layers)):
    net.layers[i].reset_parameters()

# Uncomment this and change the filename
# model_path = "../models/ff_net_mnist_d9d9f7.pt"

logger.info("Loading the saved model...")
net = suite.load(model_path)

# %% Retest the model
suite.test()
