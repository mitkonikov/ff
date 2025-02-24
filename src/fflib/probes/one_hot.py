import torch

from fflib.interfaces.iffprobe import IFFProbe

from typing import List, Callable


class TryAllClasses(IFFProbe):
    def __init__(
        self, callback: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], output_classes: int
    ):
        self.callback = callback
        self.output_classes = output_classes

    def class_goodness(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        goodness_per_label: List[torch.Tensor] = []

        for label in range(self.output_classes):
            y = torch.zeros((batch_size, self.output_classes)).to(x.device)
            y[:, label] = 1

            # self.callback should return a Tensor of size (batch_size, 1)
            result = self.callback(x, y)
            if len(result.shape) == 1:
                result = result.unsqueeze(1)

            goodness_per_label.append(result)

        # goodness_per_layer is a Tensor of size(output_classes, batch_size, 1)
        result = torch.cat(goodness_per_label, 1)  # (batch_size, output_classes)
        return result

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Goes over all possible One Hot Encodings and takes the one with the maximum goodness.
        This is achieved by asking the callback function for each possible encoding.

        Args:
            x (torch.Tensor): Tensor with shape (batch_size, features)

        Returns:
            torch.Tensor: Tensor with shape (batch_size, ) containing the predicted labels
        """

        goodness = self.class_goodness(x)
        best_label = torch.argmax(goodness, 1)  # (batch_size, )
        return best_label
