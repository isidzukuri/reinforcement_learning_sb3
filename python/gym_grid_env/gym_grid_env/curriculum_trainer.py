from agent import Agent
from agent.model_manager import ModelManager
from grid_env import GridEnv
from gym_grid_env.gym_grid_env import GymGridEnv
from gym_grid_env.end_of_curriculum_callback import EndOfCurriculumCallback
from gym_grid_env.final_plotting_callback import FinalPlottingCallback
from gym_grid_env.peak_reward_callback import PeakRewardCallback
from gym_grid_env.step_interval_progress_bar import StepIntervalProgressBar
from stable_baselines3.common.env_checker import check_env

# class CurriculumTrainer:
#     @staticmethod
#     def train_agent():
stages = [
    { 'title': 'S 10x10', 'env_size': [10, 10], 'reward_treshold': 51, 'max_steps': 200 },
    { 'title': 'SM 20x20', 'env_size': [20, 20], 'reward_treshold': -100, 'max_steps': 400 },
    { 'title': 'M 40x40', 'env_size': [40, 40], 'reward_treshold': -1000, 'max_steps': 1000 },
    { 'title': 'L 60x60', 'env_size': [60, 60], 'reward_treshold': -2000, 'max_steps': 2500 },
    # { 'title': 'XL 80x80', 'env_size': [80, 80], 'reward_treshold': 30, 'max_steps': 3000 },
]

stage = stages.pop(0)

type_mapping = {name: data['id'] for name, data in GridEnv.TYPES.items()}

agent = Agent(
    window_size=[5, 5],
    memory_grid_size=[0, 0],
    type_ids=type_mapping
)

env = GymGridEnv(agent=agent, next_stages=stages)
env.set_stage(stage)
check_env(env, warn=True)

model = ModelManager().create_new(env)
agent.model = model

total_timesteps = 100000

end_of_curriculum_callback = EndOfCurriculumCallback(env)
progress_callback = StepIntervalProgressBar(total_timesteps=total_timesteps, update_interval=1000)
final_plot_callback = FinalPlottingCallback(env=env, update_interval=1000)
peak_reward_callback = PeakRewardCallback(env)

print("Starting DQN training with Curriculum Learning")
model.learn(
    total_timesteps=total_timesteps,
    callback=[
        end_of_curriculum_callback,
        peak_reward_callback,
        progress_callback,
        final_plot_callback
    ],
    progress_bar=False
)

model.save("dqn_grid_curriculum_agent")
print("Training complete! Model saved as 'dqn_grid_curriculum_agent.zip'.")

# CurriculumTrainer.train_agent()