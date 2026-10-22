# Grid Agent

`grid-agent` provides an agent-side persistent map for partially observed
grid environments, plus inference through a Stable-Baselines3 DQN model.

The map has six `int32` channels: finish locations, visited cells, confirmed
walls, confirmed open space, active bonuses, and explored cells. Consumed
bonuses remain marked as `-1` even when they appear again in later local
observations.

## Install

```bash
python -m pip install .
```

## Usage

```python
from agent import Agent

agent = Agent(model_path="path/to/model.zip")
agent.update_memory([7, 7], local_state)
action = agent.select_action([7, 7], ["up", "down", "left", "right"])
```

`window_size` and `memory_grid_size` are `[width, height]` pairs. The
observation window must have positive odd dimensions so it can be centered on
the agent position.
