import torch
from torch import Tensor
from torch import nn


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
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

        self.embed = nn.Embedding(num_classes, num_features * 3)
        self.embed.weight.data[:, : 2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, 2 * num_features :].zero_()  # Initialise bias at 0

    def forward(self, x: Tensor, y: Tensor):
        means = torch.mean(x, dim=(2, 3))

        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)

        means = (means - m) / (torch.sqrt(v + 1e-5))

        h = self.instance_norm(x)

        gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
        h = h + means[..., None, None] * alpha[..., None, None]
        out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)

        return out


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes,
        down_sample=False,
        activation_function=nn.ELU(),
        adjust_padding=False,
        dilation=None,
    ):
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

        shortcut_padding = 0 if dilation is None else dilation
        shortcut_dilation = 1 if dilation is None else dilation

        if self.down_sample or input_dim != output_dim:
            self.shortcut = nn.Conv2d(input_dim, output_dim, 3, padding=shortcut_padding, dilation=shortcut_dilation)
        else:
            self.shortcut = nn.Identity()

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


class NCSN(nn.Module):
    def __init__(self):
        super().__init__()
