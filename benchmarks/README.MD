
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
## Run Python Pathfinder

From the repository root, run Pathfinder with the CSV grid and ONNX model in
`benchmarks/runtime/`:

```bash
run_pathfinder.py
```

The script prints the number of actions taken and the grid with the traversed
path highlighted.

## Run Rust Pathfinder

```
./pathfinder
```

## Compare python and rust implementations

Benchmarked on Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz.

Both implementations run on the same generated 15x15 grid:

![Generated 15x15 benchmark grid](runtime/grid_output.png)

The Python and Rust implementations produce the following path outputs:

| Python | Rust |
| --- | --- |
| ![Python Pathfinder output](runtime/python_pathfinder_output.jpg) | ![Rust Pathfinder output](runtime/rust_pathfinder_output.jpg) |

```
sudo apt install hyperfine
hyperfine --runs 100 \
  "./pathfinder" \
  "run_pathfinder.py"
```
results:
```
Benchmark 1: ./pathfinder
  Time (mean ± σ):      30.3 ms ±   5.3 ms    [User: 29.9 ms, System: 6.6 ms]
  Range (min … max):    26.1 ms …  59.4 ms    100 runs
 
Benchmark 2: run_pathfinder.py
  Time (mean ± σ):      2.305 s ±  0.166 s    [User: 2.256 s, System: 0.242 s]
  Range (min … max):    2.113 s …  2.928 s    100 runs
 
Summary
  './pathfinder' ran 76.06 ± 14.33 times faster than 'run_pathfinder.py'
```