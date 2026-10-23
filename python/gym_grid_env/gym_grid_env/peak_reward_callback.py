from stable_baselines3.common.callbacks import BaseCallback

from gym_grid_env.gym_grid_env import GymGridEnv


class PeakRewardCallback(BaseCallback):
    def __init__(self, env: GymGridEnv):
        super().__init__()
        self.peak_reward = float("-inf")
        self.current_stage_title = None
        self.env = env

    def _on_step(self) -> bool:
        stage_title = self.env.current_stage.get("title", "Unknown Stage")

        if stage_title != self.current_stage_title:
            self.current_stage_title = stage_title
            self.peak_reward = float("-inf")
            print(f"\n[PeakReward] Tracking new stage: {self.current_stage_title}")

        current_reward = self.env.current_episode_reward

        if current_reward > self.peak_reward:
            self.peak_reward = current_reward
            print(
                f"[{self.current_stage_title}] 🏆 New Peak Reward Reached: "
                f"{self.peak_reward}"
            )

        return True
