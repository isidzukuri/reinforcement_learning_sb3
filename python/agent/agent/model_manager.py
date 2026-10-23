from pathlib import Path
from typing import Any

import torch as th
import onnxruntime as ort
from stable_baselines3 import DQN

from agent.cnn_policy import CNNPolicy


class ModelManager:
    def _policy_kwargs(self) -> dict[str, Any]:
        return dict(
            features_extractor_class=CNNPolicy,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )

    def create_new(self, env: object) -> DQN:
        model = DQN(
            policy="CnnPolicy",
            policy_kwargs=self._policy_kwargs(),
            env=env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="./dqn_grid_tensorboard/",
        )

        return model

    
    def load_from_file(self, path_tofile: str) -> DQN:
        model_path = Path(path_tofile)
        if model_path.suffix.lower() == ".zip":
            model_path = model_path.with_suffix("")

        return DQN.load(
            str(model_path),
            custom_objects={"policy_kwargs": self._policy_kwargs()},
        )

    
    def load_onnx(self, from_path: str) -> ort.InferenceSession:
        return ort.InferenceSession(from_path)


    def export_to_onnx(self, model: DQN, export_to_path: str) -> None:
        q_network = model.policy.q_net
        was_training = q_network.training
        q_network.eval()
        try:
            dummy_input = th.zeros(
                (1, *model.observation_space.shape),
                dtype=th.float32,
                device=model.device,
            )
            th.onnx.export(
                q_network,
                dummy_input,
                export_to_path,
                input_names=["observations"],
                output_names=["q_values"],
                dynamic_axes={
                    "observations": {0: "batch_size"},
                    "q_values": {0: "batch_size"},
                },
                opset_version=17,
                dynamo=False,
            )
        finally:
            q_network.train(was_training)