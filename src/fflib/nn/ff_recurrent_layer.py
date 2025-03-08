import torch
import torch.nn as nn

from torch.nn import Module, ReLU
from torch.optim import Adam
from fflib.interfaces.iff_recurrent_layer import IFFRecurrentLayer
from typing import Callable, List, Tuple, cast, Any


class FFRecurrentLayer(IFFRecurrentLayer):
    def __init__(
        self,
        fw_features: int,
        rc_features: int,
        bw_features: int,
        loss_threshold: float,
        lr: float,
        activation_fn: Module = ReLU(),
        maximize: bool = True,
        beta: float = 0.7,
        optimizer: Callable[..., Any] = Adam,
        device: Any | None = None,
    ):
        super(FFRecurrentLayer, self).__init__()
        self.loss_threshold = loss_threshold
        self.activation_fn = activation_fn
        self.maximize = maximize
        self.beta = beta
        self.fw_features = fw_features
        self.rc_features = rc_features
        self.bw_features = bw_features

        # fw means Forward Weight
        # bw means Backward Weight
        self.fw = nn.Parameter(torch.Tensor(rc_features, fw_features).to(device))
        self.bw = nn.Parameter(torch.Tensor(rc_features, bw_features).to(device))

        # Bias for each layer
        self.fb = nn.Parameter(torch.Tensor(rc_features).to(device))

        # Setup the Optimizer
        self.opt: torch.optim.Optimizer = optimizer(self.parameters(), lr)

        # Initialize parameters
        self.reset_parameters()

    def get_dimensions(self) -> int:
        return self.rc_features

    def set_lr(self, lr: float) -> None:
        """Use this function to update the learning rate while training.

        Args:
            lr (float): New learning rate.
        """
        self.opt.param_groups[0]["lr"] = lr

    def reset_parameters(self) -> None:
        for weight in [self.fw, self.bw]:
            nn.init.orthogonal_(weight)

        if self.fb is not None:
            for bias in [self.fb]:
                nn.init.uniform_(bias)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_recc: torch.Tensor,
        x_next: torch.Tensor,
    ) -> torch.Tensor:

        # Normalization
        hf: torch.Tensor = x_prev / (x_prev.norm(2, 1, keepdim=True) + 1e-4)
        hb: torch.Tensor = x_next / (x_next.norm(2, 1, keepdim=True) + 1e-4)

        # Multiply the weights and the features in the forward and backward direction
        f = torch.mm(hf, self.fw.T)
        b = torch.mm(hb, self.bw.T)

        # Main equation
        return cast(
            torch.Tensor,
            (
                self.beta * self.activation_fn(f + b + self.fb.unsqueeze(0))
                + (1 - self.beta) * x_recc
            ),
        )

    def goodness(
        self,
        x_prev: torch.Tensor,
        x_recc: torch.Tensor,
        x_next: torch.Tensor,
        logistic_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.log(1 + torch.exp(x)),
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_prev = x_prev.clone().detach()
        x_recc = x_recc.clone().detach()
        x_next = x_next.clone().detach()

        y = self.forward(x_prev, x_recc, x_next)
        z = y.pow(2).mean(1) - self.loss_threshold
        z = -z if inverse else z
        g = logistic_fn(z)
        return g, y

    def run_train(
        self,
        h_pos: List[torch.Tensor],
        h_neg: List[torch.Tensor],
        index: int,
    ) -> None:

        g_pos = self.goodness(h_pos[index - 1], h_pos[index], h_pos[index + 1], inverse=True)[0]
        g_neg = self.goodness(h_neg[index - 1], h_neg[index], h_neg[index + 1], inverse=False)[0]

        loss = torch.cat([g_pos, g_neg]).mean()

        # Zero the gradients
        self.opt.zero_grad()

        # Compute the backward pass
        loss.backward()  # type: ignore

        # Perform a step of optimization
        self.opt.step()


class FFRecurrentLayerDummy(IFFRecurrentLayer):
    def __init__(self, dimensions: int):
        self.rc_features = dimensions

    def reset_parameters(self) -> None:
        pass

    def get_dimensions(self) -> int:
        return self.rc_features

    def goodness(
        self,
        x_prev: torch.Tensor,
        x_recc: torch.Tensor,
        x_next: torch.Tensor,
        logistic_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.log(1 + torch.exp(x)),
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x_recc, x_recc

    def run_train(
        self,
        h_pos: List[torch.Tensor],
        h_neg: List[torch.Tensor],
        index: int,
    ) -> None:
        pass
