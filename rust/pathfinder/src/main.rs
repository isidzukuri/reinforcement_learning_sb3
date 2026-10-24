fn main() {
    let executable_dir = std::env::current_exe()
        .expect("failed to locate the running executable")
        .parent()
        .expect("running executable path has no parent directory")
        .to_path_buf();
    let runtime_dir = executable_dir.join("runtime");
    let grid_file = runtime_dir.join("grid_output.csv");
    let model_file = runtime_dir.join("model.onnx");

    pathfinder::runner::call(grid_file, model_file);
}
