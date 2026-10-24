use std::path::Path;

use crate::agent::Agent;
use crate::grid_env::{CellType, CsvLoader, GridEnv, Printer};
use crate::model_loader;

pub fn call(grid_file_path: impl AsRef<Path>, model_file_path: impl AsRef<Path>) {
    let grid_env =
        CsvLoader::load(grid_file_path.as_ref()).expect("failed to load grid from CSV file");
    let model = model_loader::call(model_file_path).expect("failed to load ONNX model");
    let agent = Agent::new(
        model,
        Runner::AGENT_WINDOW_SIZE,
        [grid_env.width, grid_env.height],
    );

    let mut runner = Runner::new(grid_env, agent);

    match runner.call() {
        Ok(true) => {
            println!("Path is found after {} iterations", runner.timesteps);
            Printer::print(&runner.grid_env, Some(&runner.path));
        }
        Ok(false) => println!("Agent did not reach the finish."),
        Err(error) => eprintln!("{error}"),
    }
}

pub struct Runner {
    lock: bool,
    grid_env: GridEnv,
    agent: Agent,
    timesteps: usize,
    path: Vec<[usize; 2]>,
}

impl Runner {
    pub const AGENT_WINDOW_SIZE: [usize; 2] = [5, 5];
    pub const ACTIONS_LIMIT: usize = 2000;

    pub fn new(grid_env: GridEnv, agent: Agent) -> Self {
        Self {
            lock: false,
            grid_env,
            agent,
            timesteps: 0,
            path: Vec::new(),
        }
    }

    pub fn call(&mut self) -> Result<bool, String> {
        if self.lock {
            self.agent.reset();
        }
        self.lock = true;
        self.timesteps = 0;
        self.path.clear();

        let mut agent_pos = self.grid_env.start;
        if !Self::is_valid_position(
            &self.grid_env,
            [agent_pos[0] as isize, agent_pos[1] as isize],
        ) {
            return Err("grid start position must be in bounds and not a wall".into());
        }
        self.path.push(agent_pos);

        for _ in 0..Self::ACTIONS_LIMIT {
            if *self.grid_env.grid.get(0, agent_pos[1], agent_pos[0]) == CellType::Finish {
                return Ok(true);
            }

            self.timesteps += 1;
            self.agent
                .update_memory(agent_pos, &self.grid_env)
                .map_err(|error| format!("failed to update agent memory: {error}"))?;
            let action = self
                .agent
                .select_action(agent_pos)
                .map_err(|error| format!("failed to select an action: {error}"))?;
            let [dx, dy] = action.delta();
            let next_pos = [agent_pos[0] as isize + dx, agent_pos[1] as isize + dy];

            if Self::is_valid_position(&self.grid_env, next_pos) {
                agent_pos = [next_pos[0] as usize, next_pos[1] as usize];
                self.path.push(agent_pos);
            }
        }

        Err(format!(
            "Agent did not reach the finish within {} actions.",
            Self::ACTIONS_LIMIT
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_runner(start: [usize; 2], cell: CellType) -> Runner {
        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/runtime/model.onnx");
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

        assert_eq!(runner.call(), Ok(true));
        assert_eq!(runner.timesteps, 0);
        assert_eq!(runner.path, vec![[0, 0]]);
    }

    #[test]
    fn call_returns_error_when_start_is_out_of_bounds() {
        let mut runner = test_runner([1, 0], CellType::Finish);

        assert_eq!(
            runner.call(),
            Err("grid start position must be in bounds and not a wall".into())
        );
        assert!(runner.path.is_empty());
    }

    #[test]
    fn call_returns_error_when_start_is_a_wall() {
        let mut runner = test_runner([0, 0], CellType::Wall);

        assert_eq!(
            runner.call(),
            Err("grid start position must be in bounds and not a wall".into())
        );
    }

    #[test]
    fn repeated_call_resets_timesteps_and_path() {
        let mut runner = test_runner([0, 0], CellType::Finish);
        runner.timesteps = 12;
        runner.path.push([4, 5]);

        assert_eq!(runner.call(), Ok(true));
        assert_eq!(runner.timesteps, 0);
        assert_eq!(runner.path, vec![[0, 0]]);
    }
}
