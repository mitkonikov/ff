import pytest
import torch

from fflib.nn.ff_linear import FFLinear


def test_setup_linear():
    linear = FFLinear(in_features=10, out_features=2, loss_threshold=20, lr=0.02)

    x = torch.randn((8, 10))
    y = linear.forward(x)

    assert y.shape == (8, 2)


def test_linear_call():
    linear = FFLinear(in_features=10, out_features=2, loss_threshold=20, lr=0.02)

    x = torch.randn((8, 10))
    y1 = linear(x)
    y2 = linear.forward(x)

    assert torch.equal(y1, y2)
    assert y2.requires_grad == y1.requires_grad
    assert y2.requires_grad


def test_train_linear_basic():
    batch_size = 128

    torch.manual_seed(42)

    linear = FFLinear(in_features=10, out_features=2, loss_threshold=1, lr=0.02)

    x_train_pos = torch.cat((torch.rand((batch_size, 5)), torch.zeros((batch_size, 5))), dim=1)
    x_train_neg = torch.cat((torch.zeros((batch_size, 5)), torch.rand((batch_size, 5))), dim=1)

    x_test_pos = torch.cat((torch.rand((batch_size, 5)), torch.zeros((batch_size, 5))), dim=1)
    x_test_neg = torch.cat((torch.zeros((batch_size, 5)), torch.rand((batch_size, 5))), dim=1)

    for i in range(20):
        linear.run_train(x_train_pos, x_train_neg)

    g_pos, _ = linear.goodness(x_test_pos)
    g_neg, _ = linear.goodness(x_test_neg)

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
    # assert False
