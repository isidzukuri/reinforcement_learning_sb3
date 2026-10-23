
## Generate a 15x15 grid

From this directory, run:

```bash
python generate_grid.py
```

The script prints the grid to the terminal and saves `grid_output.png` and
`grid_output.csv`. Install the optional PNG dependency first
with `python -m pip install ".[png]"` from `python/grid_env/`.

## Export a DQN model to ONNX

From the repository root, run:

```bash
python export_model_to_onnx.py
```

This loads `dqn_cnn_model.zip` and saves `dqn_cnn_model.onnx` in this directory.
To use another model or choose a different output path, pass the model path and
`--output`:

```bash
python export_model_to_onnx.py dqn_cnn_model.zip --output runtime/model.onnx
```
## Run Pathfinder

From the repository root, run Pathfinder with the CSV grid and ONNX model in
`benchmarks/runtime/`:

```bash
/home/isidzukuri/second_ssd/python_stuff/rl/.venv/bin/python run_pathfinder.py
```

The script prints the number of actions taken and the grid with the traversed
path highlighted.
