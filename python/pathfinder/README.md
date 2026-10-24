# Pathfinder

```python
from pathfinder import Pathfinder

pathfinder = Pathfinder(
    "grid.csv",
    "model.onnx",
)

pathfinder.call()

grid = pathfinder.grid_env
agent = pathfinder.agent
print(pathfinder.path)
print(pathfinder.result)
pathfinder.print_result
```

`Pathfinder` uses the imported grid as the environment source, sets the agent
memory dimensions from the grid size, and assigns the ONNX inference session to
`agent.model`. The agent chooses actions by reading the model output for the
current observation and will keep stepping through `GridEnv.ACTIONS` until it
reaches the finish tile or the action limit is exceeded.

`call()` returns `True` once the agent reaches the finish, and stores each visited
position in `pathfinder.path`. If the agent does not reach the finish within
`Pathfinder.ACTIONS_LIMIT` actions, it raises a `RuntimeError`.
