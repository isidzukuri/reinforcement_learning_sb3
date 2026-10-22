from stable_baselines3 import DQN

from agent.cnn_policy import CNNPolicy


class ModelManager:
    def create_new(self, env: object) -> DQN:
        policy_kwargs = dict(
            features_extractor_class=CNNPolicy,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )

        model = DQN(
            policy="CnnPolicy",
            policy_kwargs=policy_kwargs,
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