import torch

from fflib.nn.ff_recurrent_layer import FFRecurrentLayer
from fflib.enums import SparsityType


def test_setup_recurrent() -> None:
    recurrent = FFRecurrentLayer(
        fw_features=10,
        rc_features=5,
        bw_features=2,
        loss_threshold=20,
        lr=0.02,
    )

    print("HOYER:", recurrent.sparsity(SparsityType.HOYER))
    print("ENTROPY_BASED:", recurrent.sparsity(SparsityType.ENTROPY_BASED))
