from fflib.utils.data.mnist import FFMNIST
from fflib.utils.data.fashion_mnist import FFFashionMNIST
from fflib.utils.data.xor import FFXOR
from fflib.utils.data.cifar10 import FFCIFAR10

from typing import Any


def CreateDatasetFromName(
    name: str,
    batch_size: int,
    validation_split: float,
    use: float = 1.0,
    **kwargs: Any,
) -> FFMNIST | FFFashionMNIST | FFXOR | FFCIFAR10 | None:
    """Create a Dataset object via a name of the dataset.
    This is used to dynamically set the dataset from CLI arguments.

    Args:
        name (str): Short name of the Dataset.
            Implemented: MNIST, FashionMNIST, XOR
        batch_size (int): Batch size
        validation_split (float): What portion of the dataset should be used for validation.
        use (float, optional): What portion of the dataset should it be used. Defaults to 1.0.
        kwargs (Dict[str, Any], optional): Other arguments. Defaults to {}.

    Returns:
        FFMNIST | FFFashionMNIST | FFXOR: Dataset object
    """
    name = name.lower()
    if name == "mnist":
        return FFMNIST(
            batch_size,
            validation_split,
            use=use,
            **kwargs,
        )
    elif name == "fashionmnist" or name == "fashion" or name == "fashion_mnist":
        return FFFashionMNIST(
            batch_size,
            validation_split,
            use=use,
            **kwargs,
        )
    elif name == "cifar10":
        return FFCIFAR10(
            batch_size,
            validation_split,
            use=use,
            **kwargs,
        )
    elif name == "xor":
        return FFXOR(
            batch_size,
            **kwargs,
        )

    return None
