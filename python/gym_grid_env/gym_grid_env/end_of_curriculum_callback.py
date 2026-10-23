from stable_baselines3.common.callbacks import BaseCallback

from gym_grid_env.gym_grid_env import GymGridEnv


class EndOfCurriculumCallback(BaseCallback):
    def __init__(self, env: GymGridEnv):
        super().__init__()
        self.env = env

    def _on_step(self) -> bool:
        if getattr(self.env, "all_stages_completed", False):
            print("All curriculum stages completed! Stopping training.")
            return False
        return True
