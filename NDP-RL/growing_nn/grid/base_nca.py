from typing import Any, Optional, Tuple  # noqa

import torch
from torch.utils import checkpoint


class BaseNCA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to_device(self, device: torch.device):
        self.device = device
        self.to(self.device)

    def generate_seed(self, num_seeds: int = 1, device: Optional[torch.device] = None):
        """
        Initial nca seed state. This is also used to initialize sample pool
        Args:
            num_seeds (int, optional): Number of seeds to generate. Defaults to 1.
        """
        raise NotImplementedError("Seed not implemented!")

    def alive(self, x):
        """
        Returns binary array denoting which cells are alive
        Args:
            x ([type]): [description]
        """
        return torch.ones_like(x, dtype=torch.bool, device=x.device)

    def get_stochastic_update_mask(self, x):
        """
        Return stochastic update mask
        Args:
            x ([type]): [description]
        """
        return torch.ones_like(x[:, 0:1], device=x.device).float()

    def update(self, x):
        raise NotImplementedError("update() is not implemented!")

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

    def grow(
        self,
        x: torch.Tensor,
        num_steps: int,
        use_checkpointing: bool = False,
        *args,
        **kwargs
    ) -> torch.Tensor:
        # for i in range(num_steps):
        #     x = checkpoint.checkpoint(self.forward, x)
        if use_checkpointing:
            return checkpoint.checkpoint_sequential([self] * num_steps, 12, x)
        for _ in range(num_steps):
            x = self.forward(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
