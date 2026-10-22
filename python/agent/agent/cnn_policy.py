from typing import Any
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: Any, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Extract shape directly from whatever observation space object SB3 passes
        obs_shape = observation_space.shape
        n_input_channels = obs_shape[0]

        # Small convolutional layers with padding to preserve spatial dimensions
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Automatically compute the flattened feature size using zeros instead of .sample()
        with th.no_grad():
            sample = th.zeros((1, *obs_shape), dtype=th.float32)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    