import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

from train import Trainer


class ScoreNetTrainer(Trainer):
    def __init__(self, train_loader: DataLoader, score_net: Module, device: torch.device = torch.device("cpu")):
        super().__init__(train_loader, ModuleList([score_net]), device)

    def _loss_batch(self, x: Tensor, loss: Module) -> Tensor:
        score_net = self.models[0]
        return loss(x, score_net)
