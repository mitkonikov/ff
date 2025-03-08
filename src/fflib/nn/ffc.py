import torch

from torch.nn import Module
from torch.optim import Adam
from fflib.interfaces.iff import IFF
from fflib.nn.ff_linear import FFLinear

from typing import List, Callable, Any


class FFC(IFF, Module):
    def __init__(
        self,
        layers: List[FFLinear],
        classifier_lr: float,
        output_classes: int = 10,
        device: Any | None = None,
    ):
        super().__init__()

        if len(layers) == 0:
            raise ValueError("FFC has to have at least one layer!")

        self.layers: List[FFLinear] = layers

        # Setup a classifier layer
        in_features = sum(layer.out_features for layer in self.layers)
        self.classifier = torch.nn.Linear(in_features, output_classes, device=device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.classifier.parameters(), classifier_lr)
        self.relu: Callable[..., torch.Tensor] = torch.nn.ReLU()

    def get_layer_count(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        for layer in self.layers:
            x = layer(x)

            if x is not None:
                activations.append(x.clone().detach())

        features = torch.cat(activations, dim=1)
        classifier_output: torch.Tensor = self.classifier(features)
        classifier_output = self.relu(classifier_output)
        return classifier_output.argmax(1)

    def run_train_combined(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> None:
        for _, layer in enumerate(self.layers):
            layer.run_train(x_pos, x_neg)

            x_pos = layer(x_pos)
            x_neg = layer(x_neg)

    def train_classifier(self, x_pos: torch.Tensor, y_pos: torch.Tensor) -> None:
        activations = []
        for layer in self.layers:
            x_pos = layer(x_pos)
            activations.append(x_pos.clone().detach())

        # Train the classifier layer with the positive data
        self.optimizer.zero_grad()
        prediction = self.classifier(torch.cat(activations, dim=1))
        loss = self.criterion(prediction, y_pos)
        loss.backward()
        self.optimizer.step()

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
