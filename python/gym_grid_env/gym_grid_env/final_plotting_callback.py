from stable_baselines3.common.callbacks import BaseCallback

from gym_grid_env.gym_grid_env import GymGridEnv

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class FinalPlottingCallback(BaseCallback):
    def __init__(self, env, update_interval: int = 100):
        super().__init__()
        # Safely unwrap SB3 vector environments to reach GymGridEnv directly
        self.env_ref = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(self.env_ref, "envs"):
            self.env_ref = self.env_ref.envs[0]
            
        self.update_interval = update_interval
        
        self.timesteps = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.len_history = []
        self.rew_history = []
        
        # Track stage transitions
        self.stage_transitions = []
        self.last_seen_stages_left = len(getattr(self.env_ref, "next_stages", []))
        print(f"[Callback Init] Initial stages left to complete: {self.last_seen_stages_left}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        # Check remaining next_stages directly from the unwrapped environment
        current_stages_left = len(getattr(self.env_ref, "next_stages", []))
        if current_stages_left != self.last_seen_stages_left:
            label = f"Stages Left: {current_stages_left}"
            print(f"[Curriculum Event] Transition detected at timestep {self.num_timesteps}! {label}")
            self.stage_transitions.append((self.num_timesteps, label))
            self.last_seen_stages_left = current_stages_left

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.len_history.append(info["episode"]["l"])
                self.rew_history.append(ep_reward)

        if self.num_timesteps % self.update_interval == 0:
            if len(self.rew_history) > 0:
                mean_reward = sum(self.rew_history[-50:]) / min(len(self.rew_history), 50)
                mean_length = sum(self.len_history[-50:]) / min(len(self.len_history), 50) if len(self.len_history) > 0 else 0

                self.timesteps.append(self.num_timesteps)
                self.ep_rewards.append(mean_reward)
                self.ep_lengths.append(mean_length)

        return True

    def _on_training_end(self) -> None:
        print(f"[Callback End] Total captured points: {len(self.timesteps)}, Total stage transitions logged: {len(self.stage_transitions)}")
        
        if len(self.timesteps) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            def plot_stage_lines(ax):
                colors = ['green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
                for idx, (t_step, label) in enumerate(self.stage_transitions):
                    color = colors[idx % len(colors)]
                    ax.axvline(x=t_step, color=color, linestyle='--', linewidth=2, alpha=0.9, label=label)

            # Panel 1: Episode Reward Mean
            ax1.plot(self.timesteps, self.ep_rewards, 'b-', marker='.', label='Recent Reward Mean')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Progress Summary with Curriculum Stages')
            ax1.grid(True)
            plot_stage_lines(ax1)
            ax1.legend(loc='upper left')

            # Panel 2: Episode Length Mean
            ax2.plot(self.timesteps, self.ep_lengths, 'r-', marker='.', label='Mean Episode Length (Steps)')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Steps')
            ax2.grid(True)
            plot_stage_lines(ax2)
            ax2.legend(loc='upper left')

            plt.tight_layout()
            plt.show()
        else:
            print("[FinalPlottingCallback] Error: No history captured. No plot to display.")
