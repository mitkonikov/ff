import pytest
import torch

from fflib.nn.ff_net import FFNet
from fflib.nn.ff_linear import FFLinear

def test_ff_net_basic():
    """
    Tests the basic functionality of a FF network with 2 layers.
    The data is randomly constructed in such a way that 
    the positive and negative data are easily distinguishable for the network.
    """
    
    batch_size = 128

    torch.manual_seed(42)
    
    layer1 = FFLinear(
        in_features=10,
        out_features=10,
        loss_threshold=1,
        lr=0.02
    )

    layer2 = FFLinear(
        in_features=10,
        out_features=2,
        loss_threshold=1,
        lr=0.02
    )

    net = FFNet([layer1, layer2], 'cpu')

    x_train_pos = torch.cat((torch.rand((batch_size, 5)), torch.zeros((batch_size, 5))), dim=1)
    x_train_neg = torch.cat((torch.zeros((batch_size, 5)), torch.rand((batch_size, 5))), dim=1)

    x_test_pos = torch.cat((torch.rand((batch_size, 5)), torch.zeros((batch_size, 5))), dim=1)
    x_test_neg = torch.cat((torch.zeros((batch_size, 5)), torch.rand((batch_size, 5))), dim=1)

    for i in range(50):
        net.run_train(x_train_pos, x_train_neg)

    g_pos = net.forward(x_test_pos)
    g_neg = net.forward(x_test_neg)

    # Expect correct shapes
    assert g_pos.shape == (batch_size,)
    assert g_neg.shape == (batch_size,)

    # Debug prints
    print(g_neg.min().item(), g_neg.max().item())
    print(g_pos.min().item(), g_pos.max().item())
    print()
    print(f"Positive mean: {g_pos.mean().item()}, Negative mean: {g_neg.mean().item()}")

    # Expect the minimum goodness of the positive data to be bigger than the max goodness of neg.
    assert g_pos.min().item() > g_neg.max().item()
    assert g_pos.mean().item() > g_neg.mean().item()
