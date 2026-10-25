# Rust Pathfinder

This crate runs a grid-world pathfinder using an ONNX model.


## Usage

```bash
cargo run -- <grid_file_path> <model_file_path>
```

For example, if the sample files are present in `runtime/`:

```bash
cargo run --release -- runtime/grid_output.csv runtime/model.onnx
```

On success, the program prints the grid with the path highlighted.


## Tests

```bash
cargo test
```