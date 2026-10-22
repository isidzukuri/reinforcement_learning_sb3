# Gym Grid Env

`gym-grid-env` adapts the `grid-env` and `grid-agent` packages to the
Gymnasium API. Observations are the agent's six-channel `int32` tensor;
actions are discrete indices into `GridEnv.ACTIONS`, in dictionary order.

## Install

```bash
python -m pip install .
```

## Usage

```python
from agent import Agent
from gym_grid_env import GymGridEnv
from grid_env import GridEnv

grid = GridEnv()
grid.generate()
agent = Agent(
    type_ids={name: data["id"] for name, data in grid.TYPES.items()},
    window_size=[3, 3],
)
env = GymGridEnv(grid, agent, max_steps=200)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

The agent starts on a randomly selected empty or bonus cell by default. Use
`env.reset(options={"agent_pos": [x, y]})` or `env.reset(initial_pos=[x, y])`
to select a specific non-wall cell.

## Testing

```bash
python -m unittest discover -s tests -v
```
