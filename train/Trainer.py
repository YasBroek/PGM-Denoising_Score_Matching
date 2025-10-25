import torch
from torch import mean
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from models.NoiseConditionalScoreNetwork import NCSN


class Trainer:
    def __init__(self, train_loader: DataLoader, score_net: NCSN, device: torch.device = torch.device("cpu")):
        self.device = device

        self._train_loader = train_loader
        self.score_net = score_net.to(device)

    def _reset_models(self):
        """
        Reset parameters of score network

        Loops through all modules in each network and calls `reset_parameters()` if the method exists.
        """
        for m in self.score_net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()  # type: ignore[operator]

    def _train_epoch(self, epoch: int, loss: Module, optimizer: Optimizer, verbose: bool):
        if verbose:
            loader = tqdm(self._train_loader, unit="batch", leave=False)
        else:
            loader = self._train_loader

        loss_e = 0
        losses = []
        n = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            loss_e = loss(x, self.score_net)

            optimizer.zero_grad()
            loss_e.backward()
            optimizer.step()

            losses.append(loss_e)
            n += 1

        if verbose:
            print(f"Epoch {epoch + 1} (Loss: {sum(losses) / n:.4f})")

    def train(self, loss: Module, optimizer: Optimizer, epochs: int = 10, verbose: bool = False):
        self._reset_models()
        self.score_net.train()

        losses = []

        for e in range(epochs):
            loss_e = self._train_epoch(e, loss, optimizer, verbose)
            losses.append(loss_e)

        return torch.as_tensor(losses)
