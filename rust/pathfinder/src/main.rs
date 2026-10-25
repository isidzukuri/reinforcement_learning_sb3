fn main() -> Result<(), pathfinder::runner::RunnerError> {
    run(std::env::args_os().collect())
}

fn run(args: Vec<std::ffi::OsString>) -> Result<(), pathfinder::runner::RunnerError> {
    if args.len() != 3 {
        let program = args
            .first()
            .map(|arg| arg.to_string_lossy())
            .unwrap_or_else(|| "pathfinder".into());

        println!("Usage: {program} <grid_file_path> <model_file_path>");
    } else {
        pathfinder::runner::call(&args[1], &args[2])?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("pathfinder_main_{unique}.csv"))
    }

    #[test]
    fn run_prints_usage_when_arguments_are_missing() {
        assert!(run(Vec::new()).is_ok());
        assert!(run(vec!["custom-pathfinder".into()]).is_ok());
    }

    #[test]
    fn run_prints_usage_when_too_many_arguments_are_given() {
        assert!(
            run(vec![
                "pathfinder".into(),
                "grid.csv".into(),
                "model.onnx".into(),
                "extra".into(),
            ])
            .is_ok()
        );
    }

    #[test]
    fn main_dispatches_process_arguments() {
        assert!(main().is_ok());
    }

    #[test]
    fn run_executes_with_grid_and_model_paths() {
        let grid_path = unique_temp_file();
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/model.onnx");
        fs::write(&grid_path, "# start: 0,0\n1\n").unwrap();

        let result = run(vec![
            "pathfinder".into(),
            grid_path.as_os_str().to_owned(),
            model_path.into_os_string(),
        ]);

        fs::remove_file(&grid_path).unwrap();
        assert!(result.is_ok());
    }
}
