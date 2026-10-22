import unittest
from unittest.mock import patch

from agent.cnn_policy import CNNPolicy
from agent.model_manager import ModelManager


class ModelManagerTests(unittest.TestCase):
    def test_create_new_builds_dqn_with_environment_and_grid_policy(self):
        env = object()
        expected_model = object()

        with patch("agent.model.DQN", return_value=expected_model) as dqn:
            model = ModelManager().create_new(env)

        self.assertIs(model, expected_model)
        args, kwargs = dqn.call_args
        self.assertEqual(args, ())
        self.assertIs(kwargs["env"], env)
        self.assertEqual(kwargs["policy"], "CnnPolicy")
        self.assertIs(kwargs["policy_kwargs"]["features_extractor_class"], CNNPolicy)
        self.assertEqual(
            kwargs["policy_kwargs"]["features_extractor_kwargs"],
            {"features_dim": 128},
        )
        self.assertFalse(kwargs["policy_kwargs"]["normalize_images"])
        self.assertEqual(
            {key: kwargs[key] for key in (
                "learning_rate",
                "buffer_size",
                "learning_starts",
                "batch_size",
                "gamma",
                "target_update_interval",
                "exploration_fraction",
                "exploration_initial_eps",
                "exploration_final_eps",
                "verbose",
                "tensorboard_log",
            )},
            {
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "gamma": 0.99,
                "target_update_interval": 500,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "verbose": 1,
                "tensorboard_log": "./dqn_grid_tensorboard/",
            },
        )


if __name__ == "__main__":
    unittest.main()
