from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torch import Tensor
from torch import nn
from torch.nn import L1Loss, MSELoss, functional as F


class AELoss(nn.Module):
    def __init__(self, alpha: float = 0, beta: float = 0):
        super().__init__()

        self.pixel_loss = MSELoss(reduction="sum")
        self.perpectual_loss = LearnedPerceptualImagePatchSimilarity(reduction="mean", normalize=True)
        self.l1_loss = L1Loss(reduction="sum")

        self.alpha = alpha
        self.beta = beta

    def _prepare_img(self, img: Tensor):
        if img.dim() == 3:
            img = img.unsqueeze(0)

        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)

        return img

    def forward(self, x: Tensor, y: Tensor):
        pix_loss = self.pixel_loss(x, y)
        l1 = self.l1_loss(x, y)
        perceptual = 0

        if x.shape[-1] >= 32 and self.beta > 0:
            x = self._prepare_img(x)
            y = self._prepare_img(y)

            perceptual = self.perpectual_loss(x, y)

        return pix_loss + self.alpha * l1 + self.beta * perceptual
