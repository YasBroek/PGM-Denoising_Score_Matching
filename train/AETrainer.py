import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

from models import Encoder, Decoder
from losses import GaussianPerturbation
from train import Trainer


class AETrainer(Trainer):
    def __init__(self, train_loader: DataLoader, encoder: Encoder, decoder: Decoder, device: torch.device = torch.device("cpu")):
        super().__init__(train_loader, ModuleList([encoder, decoder]), device)

        self.perturbation = GaussianPerturbation()

    def _loss_batch(self, x: Tensor, loss: Module) -> Tensor:
        encoder, decoder = self.models

        z = encoder(x)
        x_hat = decoder(z)

        return loss(x_hat, x)
