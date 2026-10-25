# Benchmarking and model export

This directory contains the scripts used to generate a sample grid, export the
trained DQN model to ONNX, and compare the Python and Rust pathfinding
implementations.

## Generate a 15x15 grid

From this directory, run:

```bash
python generate_grid.py
```

The script prints the grid to the terminal and saves the generated assets in
`runtime/`:

- `grid_output.csv`
- `grid_output.png`

If you want PNG output, install the optional dependency first:

```bash
python -m pip install ".[png]"
```

from `python/grid_env/`.

## Export a DQN model to ONNX

From the repository root, run:

```bash
python benchmarks/export_model_to_onnx.py
```

This loads `benchmarks/dqn_cnn_model.zip` and writes a matching ONNX file to
`benchmarks/runtime/model.onnx`.

To use a different model or custom output path:

```bash
python benchmarks/export_model_to_onnx.py benchmarks/dqn_cnn_model.zip --output benchmarks/runtime/model.onnx
```

## Run the Python pathfinder

From the repository root:

```bash
python benchmarks/run_pathfinder.py
```

The script reads the CSV grid and ONNX model from `benchmarks/runtime/` and
prints the action count and the traversed path.

## Run the Rust pathfinder

From within `benchmarks/`:

```bash
./pathfinder runtime/grid_output.csv runtime/model.onnx
```

This binary is built for the benchmark environment and reads the same runtime
files that the Python pathfinder uses.

## Compare Python and Rust implementations

Benchmarked on Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz.

Both implementations run on the same generated 15x15 grid:

![Generated 15x15 benchmark grid](runtime/grid_output.png)
Legend: 1 - finish, 2 - bonus, 3 - wall

The Python and Rust outputs are shown below:

| Python | Rust |
| --- | --- |
| ![Python Pathfinder output](runtime/python_pathfinder_output.jpg) | ![Rust Pathfinder output](runtime/rust_pathfinder_output.jpg) |


Both implementations followed the same path:

![path](runtime/grid_output_path.png)

To reproduce the timing comparison with `hyperfine`:

```bash
sudo apt install hyperfine
hyperfine --runs 100 \
  "./pathfinder runtime/grid_output.csv runtime/model.onnx" \
  "python run_pathfinder.py"
```

Example result:

```text
Benchmark 1: ./pathfinder runtime/grid_output.csv runtime/model.onnx
  Time (mean ± σ):      30.3 ms ±   5.3 ms    [User: 29.9 ms, System: 6.6 ms]
  Range (min … max):    26.1 ms …  59.4 ms    100 runs

Benchmark 2: python run_pathfinder.py
  Time (mean ± σ):      2.305 s ±  0.166 s    [User: 2.256 s, System: 0.242 s]
  Range (min … max):    2.113 s …  2.928 s    100 runs

Summary
  './pathfinder runtime/grid_output.csv runtime/model.onnx' ran 76.06 ± 14.33 times faster than 'python run_pathfinder.py'
```
