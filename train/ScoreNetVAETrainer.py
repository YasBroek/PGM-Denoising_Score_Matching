import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

from train import Trainer


class ScoreNetVAETrainer(Trainer):
    def __init__(self, train_loader: DataLoader, score_net: Module, encoder: Module, device: torch.device = torch.device("cpu")):
        super().__init__(train_loader, ModuleList([score_net]), device)

        self.encoder = encoder

    def _loss_batch(self, x: Tensor, loss: Module) -> Tensor:
        self.encoder.eval()
        score_net = self.models[0]

        mean, logvar = self.encoder(x).unbind(-1)
        sigma = torch.exp(logvar / 2)
        z = sigma * torch.randn_like(mean) + mean

        return loss(z, score_net)
