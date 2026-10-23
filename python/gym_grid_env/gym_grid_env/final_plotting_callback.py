from stable_baselines3.common.callbacks import BaseCallback

from gym_grid_env.gym_grid_env import GymGridEnv


class FinalPlottingCallback(BaseCallback):
    def __init__(self, env: GymGridEnv, update_interval: int = 100):
        super().__init__()
        self.env = env
        self.update_interval = update_interval
        self.timesteps = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.len_history = []
        self.rew_history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.len_history.append(info["episode"]["l"])
                self.rew_history.append(ep_reward)

        if self.num_timesteps % self.update_interval == 0 and self.rew_history:
            mean_reward = sum(self.rew_history[-50:]) / min(
                len(self.rew_history), 50
            )
            mean_length = (
                sum(self.len_history[-50:]) / min(len(self.len_history), 50)
                if self.len_history
                else 0
            )
            self.timesteps.append(self.num_timesteps)
            self.ep_rewards.append(mean_reward)
            self.ep_lengths.append(mean_length)

        return True

    def _on_training_end(self) -> None:
        if self.timesteps:
            import matplotlib.pyplot as plt

            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            ax1.plot(
                self.timesteps,
                self.ep_rewards,
                "b-",
                marker=".",
                label="Recent Reward Mean",
            )
            ax1.set_ylabel("Reward")
            ax1.set_title("Training Progress Summary (Final)")
            ax1.grid(True)
            ax1.legend(loc="upper left")

            ax2.plot(
                self.timesteps,
                self.ep_lengths,
                "r-",
                marker=".",
                label="Mean Episode Length (Steps)",
            )
            ax2.set_xlabel("Timesteps")
            ax2.set_ylabel("Steps")
            ax2.grid(True)
            ax2.legend(loc="upper left")

            plt.tight_layout()
            plt.show()
