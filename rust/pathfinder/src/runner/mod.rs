mod error;
mod runner;
mod solution;

pub use error::RunnerError;
pub use runner::Runner;
pub use solution::Solution;

use crate::agent::Agent;
use crate::grid_env::{CsvLoader, Printer};
use crate::model_loader;

use std::path::Path;

pub const AGENT_WINDOW_SIZE: [usize; 2] = [5, 5];

pub fn call(
    grid_file_path: impl AsRef<Path>,
    model_file_path: impl AsRef<Path>,
) -> Result<(), RunnerError> {
    let grid_env = CsvLoader::call(grid_file_path).map_err(RunnerError::CsvLoader)?;
    let model = model_loader::call(model_file_path).map_err(RunnerError::ModelLoader)?;
    let agent = Agent::new(model, AGENT_WINDOW_SIZE, [grid_env.width, grid_env.height]);

    let mut runner = Runner::new(grid_env, agent);
    let solution = runner.call()?;

    if solution.success {
        println!("Path is found after {} iterations", solution.timesteps);
        Printer::print(&runner.grid_env, Some(&solution.path));
    } else {
        println!("Agent did not reach the finish.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str, extension: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{unique}.{extension}"))
    }

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/model.onnx")
    }

    #[test]
    fn call_reports_grid_loading_errors() {
        let grid_path = unique_temp_path("runner_invalid_grid", "csv");
        fs::write(&grid_path, "not-a-cell\n").unwrap();

        let result = call(&grid_path, model_path());

        fs::remove_file(&grid_path).unwrap();
        assert!(matches!(result, Err(RunnerError::CsvLoader(_))));
    }

    #[test]
    fn call_reports_model_loading_errors() {
        let grid_path = unique_temp_path("runner_valid_grid", "csv");
        let model_path = unique_temp_path("runner_missing_model", "onnx");
        fs::write(&grid_path, "# start: 0, 0\n0\n").unwrap();

        let result = call(&grid_path, &model_path);

        fs::remove_file(&grid_path).unwrap();
        assert!(matches!(result, Err(RunnerError::ModelLoader(_))));
    }

    #[test]
    fn call_reports_when_agent_does_not_reach_finish() {
        let grid_path = unique_temp_path("runner_unfinished_grid", "csv");
        fs::write(&grid_path, "# start: 0, 0\n0\n").unwrap();

        let result = call(&grid_path, model_path());

        fs::remove_file(&grid_path).unwrap();
        assert!(result.is_ok());
    }
}
