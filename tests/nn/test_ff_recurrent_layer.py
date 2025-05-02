import torch

from fflib.nn.ff_recurrent_layer import FFRecurrentLayer
from fflib.enums import SparsityType


def test_setup_recurrent() -> None:
    torch.manual_seed(42)
    recurrent = FFRecurrentLayer(
        fw_features=10,
        rc_features=5,
        bw_features=2,
        loss_threshold=20,
        lr=0.02,
    )

    for type in SparsityType:
        t = str(type).split(".")[1]
        sparsity = recurrent.sparsity(type)["fw+bw"]
        print(f"{t}: {sparsity}")
        assert sparsity >= 0 and sparsity <= 1
