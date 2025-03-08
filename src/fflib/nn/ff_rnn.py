import torch

from torch.nn import Module
from fflib.nn.ff_recurrent_layer import FFRecurrentLayer, FFRecurrentLayerDummy
from fflib.interfaces.iff import IFF
from fflib.interfaces.iff_recurrent_layer import IFFRecurrentLayer
from typing import List, Callable, Tuple, cast, Any
from typing_extensions import Self


class FFRNN(IFF, Module):
    def __init__(
        self,
        layers: List[IFFRecurrentLayer],
        K_train: int,
        K_testlow: int,
        K_testhigh: int,
        device: Any | None = None,
    ):
        super().__init__()

        self.K_train = K_train
        self.K_testlow = K_testlow
        self.K_testhigh = K_testhigh
        self.layers = layers
        self.device = device

        # TODO: LOGS!
        # self.log: bool = False
        # self.validation_logs: List[float] = []

    @classmethod
    def from_dimensions(
        self,
        dimensions: List[int],
        K_train: int,
        K_testlow: int,
        K_testhigh: int,
        maximize: bool,  # Check if all layers have the same maximize
        activation_fn: Module,
        loss_threshold: float,
        optimizer: Callable[..., Any],
        lr: float,
        beta: float = 0.7,
        device: Any | None = None,
    ) -> Self:
        """Example wrapper of how one Forward-Forward-based Recurrent Neural Network
        should be structured. Since the FFRNN uses weights in the backward-flow direction
        in order to set all of the layers appropriately, the FFRNN wrapper requires
        a list of all of the dimensions of all of the individual dense layers.

        This wrapper currently only works with fully connected dense layers.

        The list of the dimensions should follow the following structure:
        ```py
        dimensions = [input_layer, ...recurrent_layer, output_layer]
        ```

        Args:
            dimensions (List[int]): List of the vector dimensions that each layer should have.
            activation_fn (torch.nn.Module): Activation function.
            loss_fn (Callable): Loss function.
            loss_threshold (float): Threshold dividing the positive and negative data.
            optimizer (Callable): Optimizer type.
            lr (float): Learning rate.
            K_train (int): Count of frames in training phase.
            K_testlow (int): Lowerbound frame taken into account in the testing phase.
            K_testhigh (int): Upperbound frame taken into account in the testing phase.
            maximize (bool, optional): Maximize or minimize goodness. Defaults to True.
            beta (float, optional): Beta factor. Defaults to 0.7.
            device (_type_, optional): Device. Defaults to None.
        """

        # Initialize all of the Recurrent Layers
        layers: List[IFFRecurrentLayer] = [FFRecurrentLayerDummy(dimensions[0])]
        for i in range(1, len(dimensions) - 1):
            layers.append(
                FFRecurrentLayer(
                    dimensions[i - 1],
                    dimensions[i],
                    dimensions[i + 1],
                    loss_threshold,
                    lr,
                    activation_fn,
                    maximize,
                    beta,
                    optimizer,
                    device,
                )
            )

        layers.append(FFRecurrentLayerDummy(dimensions[-1]))

        return self(
            layers,
            K_train,
            K_testlow,
            K_testhigh,
            device,
        )

    def get_layer_count(self) -> int:
        return len(self.layers) - 2

    def create_init_activations(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hp_pos = [
            torch.Tensor(
                1,
                (self.layers[i].get_dimensions()),
            ).to(self.device)
            for i in range(len(self.layers))
        ]

        hp_neg = [
            torch.Tensor(
                1,
                (self.layers[i].get_dimensions()),
            ).to(self.device)
            for i in range(len(self.layers))
        ]

        for h in hp_pos:
            torch.nn.init.uniform_(h)
        for h in hp_neg:
            torch.nn.init.uniform_(h)

        return hp_pos, hp_neg

    def _goodness_layer(
        self,
        x: List[torch.Tensor],
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """It returns the goodness along with the computed internal hidden states as the second element.
        The goodness tensor should have dimensions (BATCH_SIZE,).
        The activations tensor should have dimensions (BATCH_SIZE, LAYER_DIMENSION).

        Args:
            x (List[torch.Tensor]): Activations of all the layers at the present time
            index (int): Index of the layer from which we compute the goodness

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [goodness, activations]
        """
        return self.layers[index].goodness(x[index - 1], x[index], x[index + 1])

    def _forward_layer(self, x: List[torch.Tensor], index: int) -> torch.Tensor:
        return cast(torch.Tensor, self.layers[index].forward(x[index - 1], x[index], x[index + 1]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Setup the activations and the image input
        activations = [
            torch.Tensor(1, self.layers[i].get_dimensions()).to(self.device)
            for i in range(len(self.layers))
        ]

        for h in activations:
            torch.nn.init.uniform_(h)

        activations[0] = x.to(self.device)
        activations[-1] = y.to(self.device)

        # Push one forward pass
        for i in range(1, len(activations) - 1):
            activations[i] = self._goodness_layer(activations, i)[1].detach()

        goodness: List[torch.Tensor] = []  # (batch_size,)**layers
        # Run it for K iterations
        for frame in range(self.K_testhigh):
            new_activations = [h.clone() for h in activations]
            for i in range(1, len(self.layers) - 1):
                g, y = self._goodness_layer(activations, i)
                new_activations[i] = y.detach()

                if frame > self.K_testlow:
                    goodness.append(g)

            activations = new_activations

        # (batch_size,)**layers -> (layers, batch_size)
        result = torch.stack(goodness)
        # (layers, batch_size) -> (batch_size, )
        result = result.mean(dim=0)
        return result

    def run_train(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> None:

        # Fetch and prepare a new batch
        h_pos, h_neg = self.create_init_activations()

        # Set the input to the pos and neg data
        h_pos[0] = x_pos
        h_neg[0] = x_neg
        h_pos[-1] = y_pos
        h_neg[-1] = y_neg

        # Push one forward pass
        for i in range(1, len(self.layers) - 1):
            h_pos[i] = self._forward_layer(h_pos, i).detach()
            h_neg[i] = self._forward_layer(h_neg, i).detach()

        # Start training the network
        for _ in range(self.K_train):
            h_new_pos = [a.clone() for a in h_pos]
            h_new_neg = [a.clone() for a in h_neg]
            for i in range(1, len(self.layers) - 1):
                self.layers[i].run_train(h_pos, h_neg, i)
                h_new_pos[i] = self._forward_layer(h_pos, i).detach()
                h_new_neg[i] = self._forward_layer(h_neg, i).detach()
            h_pos = h_new_pos
            h_neg = h_new_neg

    def run_train_combined(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> None:

        raise NotImplementedError("Use run_train function with separate X and Y data inputs.")
