# %% Imports
import torch

import sys
import os
import argparse

# Get the absolute path of the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fflib.utils.data.mnist import FFMNIST, NegativeGenerator as MNISTNEG
from fflib.nn.ff_rnn import FFRNN
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.ffrnn_suite import FFRNNSuite
from fflib.utils.ff_logger import logger

# It should skip the argument parser if it finds it runs within an IPython shell.
parser = argparse.ArgumentParser(exit_on_error=False)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=60, type=int)
args = parser.parse_args(args=("" if "get_ipython" in globals() else None))

# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

torch.manual_seed(42)

# %% Setup Dataset
logger.info("Setting up MNIST dataset...")
mnist = FFMNIST(batch_size=128, validation_split=0.1, negative_generator=MNISTNEG.RANDOM)

# %% Setup the network
logger.info("Setting up the FFRNN...")

net = FFRNN.from_dimensions(
    dimensions=[784, 2000, 2000, 10],
    K_train=10,
    K_testlow=3,
    K_testhigh=8,
    maximize=True,
    activation_fn=torch.nn.ReLU(),
    loss_threshold=20,
    optimizer=torch.optim.Adam,
    lr=0.02,
    beta=0.7,
    device=device,
)
# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: net(x, y), output_classes=10)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFRNNSuite(net, probe, mnist, device)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
suite.test()

# %% Save Model
logger.info("Saving model...")
model_path = suite.save("./models/ffrnn_mnist.pt", append_hash=True)
logger.info(f"Model saved at {model_path}.")

# %% Load Model
for i in range(len(net.layers)):
    net.layers[i].reset_parameters()

# Uncomment this and change the filename
# model_path = "../models/ffrnn_mnist_d9d9f7.pt"

logger.info("Loading the saved model...")
net = suite.load(model_path)

# %% Retest the model
suite.test()
