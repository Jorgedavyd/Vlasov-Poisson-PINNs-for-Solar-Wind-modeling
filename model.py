from modulus.sym.hydra import ModulusConfig
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from lightorch.nn import DeepNeuralNetwork
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key
from typing import Dict, List, Callable
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from modulus.models.fno import FNO
import torch
from equations import EnergyConservation
from tqdm import tqdm


class PhyTrainer:
    def __init__(
        self,
        epochs: int,
        f_e: nn.Module,
        f_p: nn.Module,
        E: nn.Module,
        B: nn.Module,
        learning_rate: Dict[str, float],
        weight_decay: Dict[str, float],
        scheduler_stepsize: Dict[str, int],
        lr_lambda: Dict[str, Callable[[int], float]],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.scheduler_stepsize = scheduler_stepsize
        self.f_e = f_e
        self.f_p = f_p
        self.E = E
        self.B = B
        self.epochs: int = epochs
        self.optimizers = dict(
            f_e=Adam(
                self.f_e.parameters(),
                lr=learning_rate["f_e"],
                weight_decay=weight_decay["f_e"],
            ),
            f_p=Adam(
                self.f_p.parameters(),
                lr=learning_rate["f_p"],
                weight_decay=weight_decay["f_p"],
            ),
            E=Adam(
                self.E.parameters(),
                lr=learning_rate["E"],
                weight_decay=weight_decay["E"],
            ),
            B=Adam(
                self.B.parameters(),
                lr=learning_rate["B"],
                weight_decay=weight_decay["B"],
            ),
        )

        self.schedulers = dict(
            f_e=LambdaLR(self.optimizers["f_e"], lr_lambda["f_e"]),
            f_p=LambdaLR(
                self.optimizers["f_p"],
                lr_lambda["f_p"],
            ),
            E=LambdaLR(
                self.optimizers["E"],
                lr_lambda["E"],
            ),
            B=LambdaLR(
                self.optimizers["B"],
                lr_lambda["B"],
            ),
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pde = Residual()
        self.boundary_conditions = Dataset

    def vlasov_forward(self, input: Tensor) -> Tensor:
        return

    def training_forward(self, batch: Dict[str, Tensor]):
        continuity = ...
        vlasov = ...
        maxwell = ...

        loss *= ...
        loss.backward()
        self.optimizer.step()
        if not self.current_epoch % self.scheduler_stepsize:
            self.scheduler.step()

    def __call__(self, verbosity: bool = True) -> None:
        for self.current_epoch in range(1, self.epochs):
            for idx, batch in tqdm(enumerate(self.train_loader)):
                input: Tensor = batch["input"]  ## batch_size, time, r
                v_e_pred_t: Tensor = compute_velocity(self.f_e)
                f_e_pred_t_delta_t: Tensor = self.f_e(
                    torch.cat([input, v_e_pred_t], dim=-1)
                )
                pred: Tensor = self.f_e(input)
                loss = self.pde(pred)
