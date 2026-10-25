use std::fmt;
use std::path::Path;

use crate::agent::{Agent, AgentError};
use crate::grid_env::{CellType, CsvLoader, CsvLoaderError, GridEnv, Printer};
use crate::model_loader::{self, ModelLoaderError};

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

    match runner.call() {
        Ok(solution) => {
            if solution.success {
                println!("Path is found after {} iterations", solution.timesteps);
                Printer::print(&runner.grid_env, Some(&solution.path));
            } else {
                println!("Agent did not reach the finish.");
            }
        }
        Err(error) => return Err(error),
    }

    Ok(())
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Solution {
    pub success: bool,
    pub timesteps: usize,
    pub path: Vec<[usize; 2]>,
}

pub struct Runner {
    grid_env: GridEnv,
    agent: Agent,
}

impl Runner {
    pub const AGENT_WINDOW_SIZE: [usize; 2] = [5, 5];
    pub const ACTIONS_LIMIT: usize = 2000;

    pub fn new(grid_env: GridEnv, agent: Agent) -> Self {
        Self { grid_env, agent }
    }

    pub fn call(&mut self) -> Result<Solution, RunnerError> {
        let mut solution = Solution::default();
        let mut agent_pos = self.grid_env.start;
        if !Self::is_valid_position(
            &self.grid_env,
            [agent_pos[0] as isize, agent_pos[1] as isize],
        ) {
            return Err(RunnerError::InvalidStartPosition);
        }
        solution.path.push(agent_pos);

        for _ in 0..Self::ACTIONS_LIMIT {
            if *self.grid_env.grid.get(0, agent_pos[1], agent_pos[0]) == CellType::Finish {
                solution.success = true;
                return Ok(solution);
            }

            solution.timesteps += 1;
            self.agent.update_memory(agent_pos, &self.grid_env);
            let action = self.agent.select_action(agent_pos)?;
            let [dx, dy] = action.delta();
            let next_pos = [agent_pos[0] as isize + dx, agent_pos[1] as isize + dy];

            if Self::is_valid_position(&self.grid_env, next_pos) {
                agent_pos = [next_pos[0] as usize, next_pos[1] as usize];
                solution.path.push(agent_pos);
            }
        }

        solution.success = false;
        Ok(solution)
    }

    fn is_valid_position(grid_env: &GridEnv, position: [isize; 2]) -> bool {
        let [x, y] = position;
        x >= 0
            && x < grid_env.width as isize
            && y >= 0
            && y < grid_env.height as isize
            && *grid_env.grid.get(0, y as usize, x as usize) != CellType::Wall
    }
}

#[derive(Debug)]
pub enum RunnerError {
    Agent(AgentError),
    InvalidStartPosition,
    CsvLoader(CsvLoaderError),
    ModelLoader(ModelLoaderError),
}

// Implement std::fmt::Display
impl fmt::Display for RunnerError {
    fn fmt(&self, fm: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agent(error) => write!(fm, "{error}"),
            Self::InvalidStartPosition => {
                fm.write_str("grid start position must be in bounds and not a wall")
            }
            Self::CsvLoader(error) => write!(fm, "{error}"),
            Self::ModelLoader(error) => write!(fm, "{error}"),
        }
    }
}

// Implement std::error::Error (enables compatibility with `source()` and `dyn Error`)
impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Agent(error) => Some(error),
            Self::InvalidStartPosition => None,
            Self::CsvLoader(error) => Some(error),
            Self::ModelLoader(error) => Some(error),
        }
    }
}

impl From<AgentError> for RunnerError {
    fn from(error: AgentError) -> Self {
        Self::Agent(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_runner(start: [usize; 2], cell: CellType) -> Runner {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model.onnx");
        let model = model_loader::call(model_path).expect("repository ONNX model should load");
        let mut grid_env = GridEnv::new(1, 1);
        grid_env.start = start;
        grid_env.grid.set(0, 0, 0, cell);
        let agent = Agent::new(
            model,
            Runner::AGENT_WINDOW_SIZE,
            [grid_env.width, grid_env.height],
        );

        Runner::new(grid_env, agent)
    }

    #[test]
    fn call_succeeds_when_start_is_finish() {
        let mut runner = test_runner([0, 0], CellType::Finish);

        assert_eq!(
            runner.call().unwrap(),
            Solution {
                success: true,
                timesteps: 0,
                path: vec![[0, 0]],
            }
        );
    }

    #[test]
    fn call_returns_error_when_start_is_out_of_bounds() {
        let mut runner = test_runner([1, 0], CellType::Finish);

        assert!(matches!(
            runner.call(),
            Err(RunnerError::InvalidStartPosition)
        ));
    }

    #[test]
    fn call_returns_error_when_start_is_a_wall() {
        let mut runner = test_runner([0, 0], CellType::Wall);

        assert!(matches!(
            runner.call(),
            Err(RunnerError::InvalidStartPosition)
        ));
    }

    #[test]
    fn repeated_call_returns_independent_solutions() {
        let mut runner = test_runner([0, 0], CellType::Finish);

        let expected = Solution {
            success: true,
            timesteps: 0,
            path: vec![[0, 0]],
        };
        assert_eq!(runner.call().unwrap(), expected.clone());
        assert_eq!(runner.call().unwrap(), expected);
    }
}
