import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

from train import Trainer


class ScoreNetAETrainer(Trainer):
    def __init__(self, train_loader: DataLoader, score_net: Module, encoder: Module, z_mean, z_std, device: torch.device = torch.device("cpu")):
        super().__init__(train_loader, ModuleList([score_net]), device)

        self.encoder = encoder.to(device)
        self.encoder.eval()

        self.z_mean = z_mean[None, ...]
        self.z_std = z_std[None, ...]

    def _loss_batch(self, x: Tensor, loss: Module) -> Tensor:
        score_net = self.models[0]

        with torch.no_grad():
            z = self.encoder(x)
            z_norm = (z - self.z_mean) / self.z_std

        return loss(z_norm, score_net)
