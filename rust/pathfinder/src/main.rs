fn main() -> Result<(), pathfinder::runner::RunnerError> {
    let executable_dir = std::env::current_exe()
        .expect("failed to locate the running executable")
        .parent()
        .expect("running executable path has no parent directory")
        .to_path_buf();
    let runtime_dir = executable_dir.join("runtime");
    let grid_file_path = runtime_dir.join("grid_output.csv");
    let model_file_path = runtime_dir.join("model.onnx");

    pathfinder::runner::call(grid_file_path, model_file_path)?;

    Ok(())
}
