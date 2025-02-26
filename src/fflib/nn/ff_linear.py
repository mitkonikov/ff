import torch
from torch.nn import Linear, Module, ReLU
from torch.optim import Adam, Optimizer

from typing import Callable, Tuple, Any


class FFLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        loss_threshold: float,
        lr: float,
        activation_fn: Module = ReLU(),
        maximize: bool = True,
        optimizer: Callable[..., Any] = Adam,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        """
        Initializes an FF Linear Dense layer.
        Each layer has it's own loss function, loss threshold and optimizer (we prefer Adam)
        because it should be fully autonomous in the training phase.

        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            loss_threshold (float): Loss threshold (dividing positive and negative data)
            lr (float): Learning rate for this layer
            activation_fn (Callable, optional): Activation function for this layer. Defaults to ReLU.
            maximize (bool, optional): Whether we are maximizing or minimizing the goodness. Defaults to True. (TODO)
            optimizer (Callable, optional): Each layer has its own optimizer. Defaults to Adam.
            bias (bool, optional): Enable bias. Defaults to True.
            device (device, optional): Device. Defaults to None.
            dtype (dtype, optional): Storage Type. Defaults to None.
        """

        super().__init__(in_features, out_features, bias, device, dtype)

        self.lr = lr
        self.loss_threshold = loss_threshold
        self.maximize = maximize
        self.activation_fn = activation_fn
        self.opt: Optimizer = optimizer(self.parameters(), lr)

    def set_lr(self, lr: float) -> None:
        """Use this function to update the learning rate while training.

        Args:
            lr (float): New learning rate.
        """
        self.opt.param_groups[0]["lr"] = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the vector through the layer activating it's neurons
        X_norm = X / (X.norm() + 1e-4)
        A = activation_fn(X_norm * W^T + B)

        Note that in FF, you have to always detach the output of the layer,
        so you don't backpropagate through all of the layers.

        Args:
            x (torch.Tensor): Input samples

        Returns:
            torch.Tensor: Output samples
        """

        # Normalize the input tensor along each row
        # Each row in x_direction is a unit vector in the direction of the corresponding row in x
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)

        # Compute the linear transformation followed by the given activation
        # weight.T is 2D tensor (ex: shape (784, 500))
        # ex: (50000, 784) * (784 * 500) => (50000 * 500) + (1 * 500)
        result: torch.Tensor = self.activation_fn(
            torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        )
        return result

    def goodness(
        self,
        x: torch.Tensor,
        logistic_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.log(1 + torch.exp(x)),
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x.detach()
        y = self.forward(x)
        z = y.pow(2).mean(1) - self.loss_threshold
        z = -z if inverse else z
        g = logistic_fn(z)
        return g, y

    # train a layer
    def run_train(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> None:
        """Run a training iteration on the layer.

        Args:
            x_pos (torch.Tensor): Positive input data
            x_neg (torch.Tensor): Negative input data
        """

        # Compute the goodness for all of the positive and negative samples
        g_pos = self.goodness(x_pos, inverse=True)[0]
        g_neg = self.goodness(x_neg, inverse=False)[0]

        loss = torch.cat([g_pos, g_neg]).mean()

        # Zero the gradients
        self.opt.zero_grad()

        # Compute the backward pass
        loss.backward()  # type: ignore

        # Perform a step of optimization
        self.opt.step()
