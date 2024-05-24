from typing import Any, Optional, Tuple  # noqa

import torch
import torch.nn.functional as F
from einops import rearrange, repeat  # noqa

from growing_nn.base_nca import BaseNCA


class UpdateNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zero_bias: bool = True,
    ):
        super(UpdateNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 128, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, self.out_channels, 1, bias=False),
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.apply(init_weights)

    def forward(self, x):
        return self.out(x)


class NCA2D(BaseNCA):
    def __init__(
        self,
        # inherited
        target_shape: Tuple[int] = (3, 64, 64),
        num_hidden_channels=16,
        zero_bias=True,
        use_living_channel: bool = True,
        living_channel_dim: Optional[int] = None,
        alpha_living_threshold: float = 0.1,
        cell_fire_rate: float = 0.5,
    ):
        super(NCA2D, self).__init__()
        self.target_shape = target_shape
        self.image_size = self.target_shape[-1]  # height of image

        self.use_living_channel = use_living_channel
        self.num_target_channels = self.get_target_channels(self.target_shape)
        self.living_channel_dim = living_channel_dim
        if self.living_channel_dim is None:
            self.living_channel_dim = self.get_living_channel_dim(self.target_shape)
        self.num_hidden_channels = num_hidden_channels
        self.num_channels = (
            self.living_channel_dim + self.num_hidden_channels + 1
        )  # output_channels + hidden channels + living_channel_dim

        self.cell_fire_rate = cell_fire_rate
        self.alpha_living_threshold = alpha_living_threshold
        self.zero_bias = zero_bias

        self.setup_network()

    def get_target_channels(self, target_shape):
        return self.target_shape[0]

    def get_living_channel_dim(self, target_shape):
        num_output_channels = self.target_shape[
            0
        ]  # by default make living channel dim the last target dim
        return num_output_channels

    def setup_network(self):
        # setup network
        self.perception_net = torch.nn.Conv2d(
            self.num_channels,
            self.num_channels * 3,
            3,
            stride=1,
            padding=1,
            groups=self.num_channels,
            bias=False,
        )
        self.update_net = UpdateNet(
            self.perception_net.out_channels, self.num_channels, self.zero_bias
        )

    def generate_seed(self, num_seeds, device: Optional[torch.device] = None):
        if device is not None:
            device = torch.device("cpu")
        seed = torch.zeros(
            num_seeds,
            self.num_channels,
            self.image_size,
            self.image_size,
            device=device,
        )
        seed[
            :, self.living_channel_dim :, self.image_size // 2, self.image_size // 2
        ] = 1.0  # rgb=0, alpha=1 = black
        return seed

    def alive(self, x):
        if not self.use_living_channel:
            return torch.ones_like(x, dtype=torch.bool, device=x.device)
        return (
            F.max_pool2d(
                x[:, self.living_channel_dim : self.living_channel_dim + 1, :, :],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > self.alpha_living_threshold
        )

    def get_stochastic_update_mask(self, x):
        """
        Return stochastic update mask
        Args:
            x ([type]): [description]
        """
        return (
            torch.clamp(torch.rand_like(x[:, 0:1], device=x.device), 0.0, 1.0).float()
            < self.cell_fire_rate
        ).float()

    def update(self, x):
        return self.update_net(self.perception_net(x))

    def forward(self, x):
        pre_life_mask = self.alive(x)

        out = self.update(x)
        rand_mask = self.get_stochastic_update_mask(x)
        out = out * rand_mask
        x = x + out

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x
