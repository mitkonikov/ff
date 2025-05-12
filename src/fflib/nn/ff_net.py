import torch

from torch.nn import Module
from fflib.interfaces.iff import IFF
from fflib.nn.ff_linear import FFLinear
from fflib.enums import SparsityType
from typing import List, Dict, Any


class FFNet(IFF, Module):
    def __init__(
        self,
        layers: List[FFLinear],
        device: Any | None,
        maximize: bool = True,
        layer_range: range | None = None,
    ):
        """Primitive MLP that uses the Forward-Forward algorithm for training.
        Each layer is being trained individually.

        By default the network is maximizing the goodness of the layers for positive data,
        but with the maximize boolean you can override the layers to minimize the goodness for positive data.
        Note: Make sure when probing a network that is trained with minimization,
        the probe is also set to target minimization.

        The "layer_range" parameter is used to specify the range of layers
        from which the goodness is taken for prediction.
        By default, it's set to None and the goodness is taken from all layers.

        Args:
            layers (List[FFLinear]): List of FF Linear Layers
            device (Any | None): Device
            maximize (bool, optional): Target optimization of goodness. Defaults to True.
            layer_range (range | None, optional):
                Range of layers from which the goodness is taken for the prediction. Defaults to None.

        Raises:
            ValueError: FFNet has to have at least one layer!
            ValueError: Layer Range has to be either None or range object.
            ValueError: Layer Range has to include at least one layer.
            ValueError: Invalid Layer Range!
        """

        super().__init__()

        if len(layers) == 0:
            raise ValueError("FFNet has to have at least one layer!")

        if layer_range is not None and not isinstance(layer_range, range):
            raise ValueError("Layer Range has to be either None or range object.")

        if isinstance(layer_range, range):
            if len(layer_range) == 0:
                raise ValueError("Layer Range has to include at least one layer.")
            if layer_range[0] < 0 or layer_range[-1] >= len(layers):
                raise ValueError("Invalid Layer Range!")

        self.device = device
        self.layers: List[FFLinear] = layers
        self.maximize: bool = maximize
        self.layer_range = layer_range

        for i in range(len(layers)):
            self.layers[i].maximize = self.maximize
            self.add_module(f"layer_{i}", layers[i])

        self._create_hooks_dict()

    def get_layer_count(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: List[torch.Tensor] = []  # (layer, batch_size, goodness)
        for i, layer in enumerate(self.layers):
            # Each layer's inference returns the goodness of the layer
            # and the output of the layer to be passed to the next
            g, x = layer.goodness(x)

            self._call_hooks("layer_activation", x, i)
            self._call_hooks("layer_goodness", g, i)

            if g is not None and (self.layer_range is None or i in self.layer_range):
                result.append(g)

        return torch.sum(torch.stack(result), dim=0)

    def run_train_combined(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> None:

        # For each layer in the neural network
        for _, layer in enumerate(self.layers):
            layer.run_train(x_pos, x_neg)

            x_pos = layer(x_pos)
            x_neg = layer(x_neg)

    def run_train(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> None:

        raise NotImplementedError(
            "Use run_train_combined in conjunction with the FFDataProcessor's combine_to_input method."
        )

    def strip_down(self) -> None:
        for layer in self.layers:
            layer.strip_down()
        delattr(self, "hooks")

    def sparsity(self, type: SparsityType) -> Dict[str, float]:
        return {
            f"layer_{i}": float(layer.sparsity(type).item()) for i, layer in enumerate(self.layers)
        }

    def stats(self) -> Dict[str, Any]:
        return {f"layer_{i}": layer.stats() for i, layer in enumerate(self.layers)}
