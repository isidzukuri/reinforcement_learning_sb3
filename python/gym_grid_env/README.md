# Gym Grid Env

`gym-grid-env` adapts the `grid-env` and `grid-agent` packages to the
Gymnasium API. Observations are the agent's six-channel `int32` tensor;
actions are discrete indices into `GridEnv.ACTIONS`.

## Testing

```bash
python -m unittest discover -s tests -v
```
