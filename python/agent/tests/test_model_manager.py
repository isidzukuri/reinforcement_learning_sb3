import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch as th

from agent.cnn_policy import CNNPolicy
from agent.model_manager import ModelManager


class ModelManagerTests(unittest.TestCase):
    def test_load_onnx_returns_inference_session(self):
        expected_session = object()

        with patch(
            "agent.model_manager.ort.InferenceSession",
            return_value=expected_session,
        ) as inference_session:
            session = ModelManager().load_onnx("model.onnx")

        self.assertIs(session, expected_session)
        inference_session.assert_called_once_with("model.onnx")

    def test_export_to_onnx_exports_q_network_with_dynamic_batch_axis(self):
        q_network = th.nn.Linear(150, 4)
        model = SimpleNamespace(
            policy=SimpleNamespace(q_net=q_network),
            observation_space=SimpleNamespace(shape=(6, 5, 5)),
            device=th.device("cpu"),
        )

        with patch("agent.model_manager.th.onnx.export") as export:
            ModelManager().export_to_onnx(model, "model.onnx")

        args, kwargs = export.call_args
        self.assertIs(args[0], q_network)
        self.assertEqual(tuple(args[1].shape), (1, 6, 5, 5))
        self.assertEqual(args[2], "model.onnx")
        self.assertEqual(kwargs["input_names"], ["observations"])
        self.assertEqual(kwargs["output_names"], ["q_values"])
        self.assertEqual(
            kwargs["dynamic_axes"],
            {
                "observations": {0: "batch_size"},
                "q_values": {0: "batch_size"},
            },
        )

    def test_load_from_file_loads_and_returns_dqn(self):
        expected_model = object()

        with patch("agent.model_manager.DQN.load", return_value=expected_model) as load:
            model = ModelManager().load_from_file("saved_model.zip")

        self.assertIs(model, expected_model)
        load.assert_called_once_with(
            "saved_model",
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": CNNPolicy,
                    "features_extractor_kwargs": {"features_dim": 128},
                    "normalize_images": False,
                },
            },
        )

    def test_create_new_builds_dqn_with_environment_and_grid_policy(self):
        env = object()
        expected_model = object()

        with patch("agent.model_manager.DQN", return_value=expected_model) as dqn:
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
