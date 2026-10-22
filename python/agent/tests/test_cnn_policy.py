import unittest

import gymnasium as gym
import numpy as np
import torch as th

from agent.cnn_policy import CNNPolicy


class CNNPolicyTests(unittest.TestCase):
    def test_forward_returns_non_negative_features_for_batch(self):
        features_dim = 64
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(6, 5, 5),
            dtype=np.float32,
        )
        policy = CNNPolicy(observation_space, features_dim=features_dim)
        observations = th.rand(4, *observation_space.shape)

        features = policy(observations)

        self.assertEqual(features.shape, (4, features_dim))
        self.assertTrue(th.isfinite(features).all())
        self.assertTrue((features >= 0).all())

    def test_default_features_dimension(self):
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(6, 5, 5),
            dtype=np.float32,
        )

        policy = CNNPolicy(observation_space)
        features = policy(th.rand(1, *observation_space.shape))

        self.assertEqual(policy.features_dim, 128)
        self.assertEqual(features.shape, (1, 128))


if __name__ == "__main__":
    unittest.main()
