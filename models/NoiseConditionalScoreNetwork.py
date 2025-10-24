from types import NoneType

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from utils import ConditionalSequential


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()

        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            )

    def forward(self, inputs: Tensor):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return output


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()

        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, : 2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features :].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes: int,
        down_sample=False,
        activation_function: nn.Module = nn.ELU(),
        dilation=None,
        adjust_padding=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.down_sample = down_sample

        conv_padding = 1 if dilation is None else dilation
        conv_dilation = 1 if dilation is None else dilation
        conv1_output_dim = input_dim if self.down_sample else output_dim

        self.normalize1 = ConditionalInstanceNorm2dPlus(input_dim, num_classes)
        self.conv1 = nn.Conv2d(input_dim, conv1_output_dim, 3, padding=conv_padding, dilation=conv_dilation)

        self.normalize2 = ConditionalInstanceNorm2dPlus(conv1_output_dim, num_classes)
        self.conv2 = nn.Conv2d(conv1_output_dim, output_dim, 3, padding=conv_padding, dilation=conv_dilation)

        self.shortcut = nn.Identity()

        if self.down_sample or input_dim != output_dim:
            shortcut_padding = 0 if dilation is None else dilation
            shortcut_dilation = 1 if dilation is None else dilation

            self.shortcut = nn.Conv2d(input_dim, output_dim, 3, padding=shortcut_padding, dilation=shortcut_dilation)

        if self.down_sample and dilation is None:
            self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
            self.shortcut = ConvMeanPool(input_dim, output_dim, adjust_padding=adjust_padding)

    def forward(self, x: Tensor, y: Tensor):
        output = self.normalize1(x, y)
        output = self.activation_function(output)
        output = self.conv1(output)

        output = self.normalize2(output, y)
        output = self.activation_function(output)
        output = self.conv2(output)

        shortcut = self.shortcut(x)

        return shortcut + output


class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, activation_function=nn.ReLU()):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = ConditionalSequential()

            for __ in range(n_stages):
                stage = ConditionalSequential(
                    ConditionalInstanceNorm2dPlus(features, num_classes), activation_function, nn.Conv2d(features, features, 3, padding=1)
                )

                block.append(stage)

            self.blocks.append(block)

    def forward(self, x, y):
        for block in self.blocks:
            residual = x
            x = block(x, y)
            x += residual

        return x


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes: list | tuple, features, num_classes):
        super().__init__()

        self.num_planes = len(in_planes)
        self.features = features

        self.blocks = nn.ModuleList()

        for i in range(self.num_planes):
            block = ConditionalSequential(
                ConditionalInstanceNorm2dPlus(in_planes[i], num_classes), nn.Conv2d(in_planes[i], features, 3, padding=1, bias=True)
            )
            self.blocks.append(block)

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)

        for i in range(self.num_planes):
            h = self.blocks[i](xs[i], y)
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h

        return sums


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages: int, num_classes, activation_function=nn.ReLU()):
        super().__init__()

        self.activation_function = activation_function
        self.stages = nn.ModuleList()

        for _ in range(n_stages):
            stage = ConditionalSequential(
                ConditionalInstanceNorm2dPlus(features, num_classes),
                nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(features, features, 3, padding=1),
            )
            self.stages.append(stage)

    def forward(self, x, y):
        x = self.activation_function(x)
        path = x

        for stage in self.stages:
            path = stage(path, y)
            x = path + x

        return x


class CondRefineBlock(nn.Module):
    def __init__(self, in_planes: list | tuple, features, num_classes, activation_function, start=False, end=False):
        super().__init__()

        self.n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(self.n_blocks):
            self.adapt_convs.append(CondRCUBlock(in_planes[i], 2, 2, num_classes, activation_function))

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, activation_function)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes)

        self.crp = CondCRPBlock(features, 2, num_classes, activation_function)

    def forward(self, xs: list | tuple, y, output_shape):
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class NCSN(nn.Module):
    def __init__(
        self,
        input_dim: torch.Size | tuple[int, int, int],
        num_cls: int,
        activation_function: nn.Module | NoneType = None,
        filters: int = 128,
    ):
        super().__init__()

        if activation_function is None:
            activation_function = nn.ELU()

        input_width = input_dim[0]
        input_channels = input_dim[2]

        self.begin_conv = nn.Conv2d(input_channels, filters, 3, padding=1)

        self.residual_block = ConditionalSequential(
            ConditionalResidualBlock(filters, filters, num_cls, False, activation_function),
            ConditionalResidualBlock(filters, filters, num_cls, False, activation_function),

            ConditionalResidualBlock(filters, 2 * filters, num_cls, True, activation_function),
            ConditionalResidualBlock(2 * filters, 2 * filters, num_cls, False, activation_function),

            ConditionalResidualBlock(2 * filters, 2 * filters, num_cls, True, activation_function, 2),
            ConditionalResidualBlock(2 * filters, 2 * filters, num_cls, False, activation_function, 2),

            ConditionalResidualBlock(2 * filters, 2 * filters, num_cls, True, activation_function, 4, input_width == 28),
            ConditionalResidualBlock(2 * filters, 2 * filters, num_cls, False, activation_function, 4),
        )

        self.refine_block = nn.ModuleList(
            [
                CondRefineBlock([2 * filters], 2 * filters, num_cls, activation_function, start=True),
                CondRefineBlock([2 * filters, 2 * filters], 2 * filters, num_cls, activation_function),
                CondRefineBlock([2 * filters, 2 * filters], filters, num_cls, activation_function),
                CondRefineBlock([filters, filters], filters, num_cls, activation_function, end=True),
            ]
        )

        self.final_block = ConditionalSequential(
            ConditionalInstanceNorm2dPlus(filters, num_cls),
            activation_function,
            nn.Conv2d(filters, input_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, y: Tensor):
        output = self.begin_conv(x)
        output = self.residual_block(output, y)

        ref = None
        for i in range(len(self.refine_block)):
            refine = self.refine_block[i]
            k_res = len(self.residual_block) - 2 * i - 1

            input_refine = [self.residual_block.outputs[k_res]]
            if ref is not None:
                input_refine.append(ref)

            shape_refine = self.residual_block.shapes[k_res][2:]

            ref = refine(input_refine, y, shape_refine)

        output = ref
        output = self.final_block(output, y)

        return output
