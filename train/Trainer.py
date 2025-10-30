import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, train_loader: DataLoader, models: ModuleList, device: torch.device = torch.device("cpu")):
        self.device = device
        self._train_loader = train_loader

        self.models = [model.to(device) for model in models]

    def _reset_models(self):
        """
        Reset parameters of score network

        Loops through all modules in each network and calls `reset_parameters()` if the method exists.
        """
        for model in self.models:
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()  # type: ignore[operator]

    def _loss_batch(self, x: Tensor, loss: Module) -> Tensor:
        raise NotImplementedError()

    def _train_epoch(self, epoch: int, loss: Module, optimizer: Optimizer, verbose: bool):
        if verbose:
            loader = tqdm(self._train_loader, unit="batch", leave=False)
        else:
            loader = self._train_loader

        loss_e = 0
        losses = []

        for x in loader:
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            x = x.to(self.device)

            loss_e = self._loss_batch(x, loss)

            optimizer.zero_grad()
            loss_e.backward()
            optimizer.step()

            losses.append(loss_e.item())

        total_loss = sum(losses) / len(losses)

        if verbose:
            print(f"Epoch {epoch + 1} (Loss: {total_loss:.4f})")

        return total_loss

    def train(self, loss: Module, optimizer: Optimizer, epochs: int = 10, verbose: bool = False, reset: bool = True):
        if reset:
            self._reset_models()

        for model in self.models:
            model.train()

        losses = []

        for e in range(epochs):
            loss_e = self._train_epoch(e, loss, optimizer, verbose)
            losses.append(loss_e)

        return torch.tensor(losses, device=self.device)
