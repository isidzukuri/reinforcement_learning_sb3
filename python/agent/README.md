# Grid Agent

`grid-agent` provides persistent map memory for a partially observed grid
environment and DQN inference with Stable-Baselines3. The package also includes
a CNN feature extractor and a helper for creating a DQN model.

## Install

From this directory, install the package and its dependencies:

```bash
python -m pip install .
```

## Agent API

Create an agent with the environment's cell-type IDs and dimensions. Positions
use `[x, y]` coordinates; `window_size` and `memory_grid_size` are
`[width, height]` pairs. The observation window must have positive odd
dimensions so it can be centered on the agent.

```python
from agent import Agent

type_ids = {"empty": 0, "finish": 1, "bonus": 2, "wall": 3}
agent = Agent(
    type_ids=type_ids,
    window_size=[5, 5],
    memory_grid_size=[15, 15],
)
agent.load_model("path/to/model.zip")

# local_state is a height-by-width grid of cell-type IDs, centered on [x, y].
agent.update_memory([7, 7], local_state)
action = agent.select_action([7, 7], ["up", "down", "left", "right"])
```

`select_action` uses deterministic model predictions. The order of
`action_list` must match the action indices used when training the model.
Calling it before `load_model` raises `RuntimeError`.

`update_memory` stores each observation in a six-channel `int32` map, in this
order: finish locations, visited cells, confirmed walls, traversable cells,
active bonuses, and revealed cells. `get_observation_tensor(agent_pos)` returns
the local view of that map with shape `(6, window_height, window_width)`.
Positions outside the memory grid are treated as walls in the observation.
`consume_bonus(pos)` marks a known active bonus as consumed and returns whether
one was present. `reset()` clears the map.

## Creating a DQN model

`ModelManager.create_new` creates an untrained Stable-Baselines3 DQN model for
an environment. Its CNN feature extractor expects observations with the same
channel-first grid shape returned by `get_observation_tensor`.
`ModelManager.load_from_file(path_tofile)` loads a saved DQN model and returns
it. `ModelManager.export_to_onnx(model, export_to_path)` exports the model's
Q-network to ONNX; the ONNX output contains Q-values for each action, so select
an action with `argmax`. `ModelManager.load_onnx(from_path)` loads the exported
file and returns an ONNX Runtime inference session.

```python
from agent import ModelManager

model = ModelManager().create_new(env)
model.learn(total_timesteps=100_000)
model.save("path/to/model")
loaded_model = ModelManager().load_from_file("path/to/model.zip")
ModelManager().export_to_onnx(loaded_model, "path/to/model.onnx")
onnx_session = ModelManager().load_onnx("path/to/model.onnx")
```

Load the saved model with `agent.load_model("path/to/model.zip")` before
selecting actions.


## Testing
```
python -m unittest discover -s tests -v
```