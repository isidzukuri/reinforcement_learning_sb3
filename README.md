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

## Rust crate

The `rust/pathfinder` crate exposes Rust modules for CSV-backed grid
environments, ONNX model loading, and agent memory/inference. Its tests can be
run from the repository root with:

```bash
cargo test --manifest-path rust/pathfinder/Cargo.toml
```

The agent expects a six-layer observation and a positive, odd `[width, height]`
window matching the ONNX model's input dimensions:

```rust
use std::collections::HashMap;
use pathfinder::agent::Agent;
use pathfinder::grid_env::{grid_env_from_csv, print_to_stdout};

let types = HashMap::from([
    ("empty".to_string(), 0),
    ("finish".to_string(), 1),
    ("bonus".to_string(), 2),
    ("wall".to_string(), 3),
]);
let mut agent = Agent::new(types, [5, 5], [15, 15])?;
agent.load_model("benchmarks/runtime/model.onnx")?;
let actions = ["top", "bottom", "left", "right"].map(str::to_string);
let action = agent.select_action([7, 7], &actions)?;

let grid = grid_env_from_csv("benchmarks/runtime/grid_output.csv")?;
print_to_stdout(&grid);
```
