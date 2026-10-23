# RL Python Packages

This repository contains the `agent`, `grid_env`, and `gym_grid_env` Python
packages under `python/`.

## Running all tests

From the repository root, run each package's `unittest` suite with:

```bash
export PYTHONPATH=python/agent:python/grid_env:python/gym_grid_env
for dir in python/*/tests; do
  .venv/bin/python -m unittest discover -s "$dir" || exit 1
done
```

This uses the repository virtual environment at `.venv` and stops if a suite
fails.
