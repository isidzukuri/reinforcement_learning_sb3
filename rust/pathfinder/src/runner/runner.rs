use super::{RunnerError, Solution};
use crate::agent::Agent;
use crate::grid_env::{CellType, GridEnv};

pub struct Runner {
    pub grid_env: GridEnv,
    agent: Agent,
}

impl Runner {
    const ACTIONS_LIMIT: usize = 2000;

    pub fn new(grid_env: GridEnv, agent: Agent) -> Self {
        Self { grid_env, agent }
    }

    pub fn call(&mut self) -> Result<Solution, RunnerError> {
        let mut solution = Solution::default();
        let mut agent_pos = self.grid_env.start;

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

            if self.is_valid_position(next_pos) {
                agent_pos = [next_pos[0] as usize, next_pos[1] as usize];
                solution.path.push(agent_pos);
            }
        }

        solution.success = false;
        Ok(solution)
    }

    fn is_valid_position(&self, position: [isize; 2]) -> bool {
        let [x, y] = position;

        if x < 0 || y < 0 {
            return false;
        }

        let x = x as usize;
        let y = y as usize;

        x < self.grid_env.width
            && y < self.grid_env.height
            && *self.grid_env.grid.get(0, y, x) != CellType::Wall
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader;
    use std::path::PathBuf;

    fn test_runner(start: [usize; 2], cell: CellType) -> Runner {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model.onnx");
        let model = model_loader::call(model_path).expect("repository ONNX model should load");
        let mut grid_env = GridEnv::new(1, 1);
        grid_env.start = start;
        grid_env.grid.set(0, 0, 0, cell);
        let agent = Agent::new(model, [5, 5], [grid_env.width, grid_env.height]);

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
