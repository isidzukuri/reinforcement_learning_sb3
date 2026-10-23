from stable_baselines3.common.callbacks import BaseCallback


class StepIntervalProgressBar(BaseCallback):
    def __init__(self, total_timesteps: int, update_interval: int = 100):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.pbar = None

    def _on_training_start(self) -> None:
        from tqdm.auto import tqdm

        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if (
            self.num_timesteps % self.update_interval == 0
            or self.num_timesteps == self.total_timesteps
        ):
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.n = self.total_timesteps
            self.pbar.refresh()
            self.pbar.close()
