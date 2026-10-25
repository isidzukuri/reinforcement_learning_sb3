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

pub fn call(
    grid_file_path: impl AsRef<Path>,
    model_file_path: impl AsRef<Path>,
) -> Result<(), RunnerError> {
    let grid_env = CsvLoader::call(grid_file_path.as_ref()).map_err(RunnerError::CsvLoader)?;
    let model = model_loader::call(model_file_path).map_err(RunnerError::ModelLoader)?;
    let agent = Agent::new(
        model,
        Runner::AGENT_WINDOW_SIZE,
        [grid_env.width, grid_env.height],
    );

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
