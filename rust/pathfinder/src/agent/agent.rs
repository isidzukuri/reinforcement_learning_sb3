use ort::session::Session as Model;
use ort::value::Tensor as OrtTensor;

use super::{AgentError, TensorLayer};
use crate::grid_env::{Action, CellType, GridEnv};
use crate::tensor::Tensor;

pub struct Agent {
    model: Model,
    window_size: [usize; 2],
    known_env: Tensor<f32>,
}

impl Agent {
    pub fn new(model: Model, window_size: [usize; 2], memory_grid_size: [usize; 2]) -> Self {
        let known_env = Tensor::new(TensorLayer::len(), memory_grid_size[1], memory_grid_size[0]);

        Self {
            model,
            window_size,
            known_env,
        }
    }

    /// Resets the agent's memory of the environment
    pub fn reset(&mut self) {
        self.known_env.fill(0.0);
    }

    /// Updates the agent's memory directly from the GridEnv.
    pub fn update_memory(&mut self, agent_pos: [usize; 2], grid_env: &GridEnv) {
        let agent_x = agent_pos[0] as isize;
        let agent_y = agent_pos[1] as isize;

        let memory_width = self.known_env.width() as isize;
        let memory_height = self.known_env.height() as isize;

        // Mark current agent position as visited in the Visited layer
        if agent_x >= 0 && agent_x < memory_width && agent_y >= 0 && agent_y < memory_height {
            self.known_env.set(
                TensorLayer::Visited as usize,
                agent_y as usize,
                agent_x as usize,
                1.0,
            );
        }

        grid_env.for_each_in_window(
            agent_pos,
            self.window_size,
            |_local_x, _local_y, grid_x, grid_y, cell_type| {
                if grid_x < 0 || grid_x >= memory_width || grid_y < 0 || grid_y >= memory_height {
                    return;
                }
                let gx = grid_x as usize;
                let gy = grid_y as usize;

                // Update Finish layer
                if cell_type == CellType::Finish {
                    self.known_env
                        .set(TensorLayer::Finish as usize, gy, gx, 1.0);
                }

                // Update Wall layer
                if cell_type == CellType::Wall {
                    self.known_env.set(TensorLayer::Wall as usize, gy, gx, 1.0);
                }

                // Update Traversable layer
                if cell_type == CellType::Empty || cell_type == CellType::Bonus {
                    self.known_env
                        .set(TensorLayer::Traversable as usize, gy, gx, 1.0);
                }

                // Update Bonus layer (respecting special -1 lock)
                let current_bonus = *self.known_env.get(TensorLayer::Bonus as usize, gy, gx);
                if current_bonus != -1.0 {
                    if cell_type == CellType::Bonus {
                        self.known_env.set(TensorLayer::Bonus as usize, gy, gx, 1.0);
                    } else {
                        self.known_env.set(TensorLayer::Bonus as usize, gy, gx, 0.0);
                    }
                }

                // Update Revealed layer (Fog of War)
                self.known_env
                    .set(TensorLayer::Revealed as usize, gy, gx, 1.0);
            },
        );
    }

    /// Chooses the next action by building the observation tensor for the agent's
    /// current position, running the model inference, and returning the action with
    /// the highest predicted value.
    pub fn select_action(&mut self, agent_pos: [usize; 2]) -> Result<Action, AgentError> {
        let obs_tensor = self.get_observation_tensor(agent_pos)?;
        let outputs = self
            .model
            .run(ort::inputs![obs_tensor])
            .map_err(AgentError::Inference)?;

        let output = outputs
            .values()
            .next()
            .ok_or(AgentError::MissingModelOutput)?;
        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(AgentError::OutputTensor)?;

        Self::action_from_output(data)
    }

    fn action_from_output(data: &[f32]) -> Result<Action, AgentError> {
        let action_index = data
            .iter()
            .enumerate()
            .filter(|&(_, &v)| !v.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or(AgentError::NoValidActionValues)?;

        let action = Action::ALL
            .get(action_index)
            .copied()
            .ok_or(AgentError::InvalidActionIndex(action_index))?;

        Ok(action)
    }

    /// Extracts the local observation window tensor around the agent's position.
    pub fn get_observation_tensor(
        &self,
        agent_pos: [usize; 2],
    ) -> Result<OrtTensor<f32>, AgentError> {
        let agent_x = agent_pos[0] as isize;
        let agent_y = agent_pos[1] as isize;

        let window_width = self.window_size[0];
        let window_height = self.window_size[1];
        let half_window_width = (window_width as isize) / 2;
        let half_window_height = (window_height as isize) / 2;

        let memory_width = self.known_env.width() as isize;
        let memory_height = self.known_env.height() as isize;

        let num_channels = self.known_env.channels();
        let mut tensor_data = Tensor::new(num_channels, window_height, window_width);
        tensor_data.fill_channel(TensorLayer::Wall as usize, 1.0);

        let start_grid_x = agent_x - half_window_width;
        let end_grid_x = start_grid_x + (window_width as isize) - 1;

        let clip_start_x = start_grid_x.max(0);
        let clip_end_x = end_grid_x.min(memory_width - 1);

        let length = (clip_end_x - clip_start_x + 1) as usize;
        let dx_local_start = (clip_start_x - start_grid_x) as usize;
        let gx_start = clip_start_x as usize;
        let env_width = self.known_env.width();

        for dy_local in 0..window_height {
            let grid_y = agent_y + (dy_local as isize) - half_window_height;

            if !(grid_y >= 0 && grid_y < memory_height) {
                continue;
            }

            let gy = grid_y as usize;
            let src_start = gy * env_width + gx_start;
            let dst_start = dy_local * window_width + dx_local_start;

            for channel_idx in 0..num_channels {
                let src_slice = &self.known_env.channel(channel_idx)[src_start..src_start + length];
                let dst_slice =
                    &mut tensor_data.channel_mut(channel_idx)[dst_start..dst_start + length];

                dst_slice.copy_from_slice(src_slice);
            }
        }

        let shape = [1, num_channels, window_height, window_width];
        let tensor = OrtTensor::from_array((shape, tensor_data.into_vec()))
            .map_err(AgentError::ObservationTensor)?;

        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::call;
    use std::path::Path;

    fn test_agent(window_size: [usize; 2], memory_grid_size: [usize; 2]) -> Agent {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/model.onnx");
        let model = call(model_path).expect("repository ONNX model should load");
        Agent::new(model, window_size, memory_grid_size)
    }

    fn test_grid() -> GridEnv {
        let mut grid = GridEnv::new(5, 5);
        let cells = [
            (1, 1, CellType::Empty),
            (2, 1, CellType::Finish),
            (3, 1, CellType::Bonus),
            (1, 2, CellType::Wall),
            (2, 2, CellType::Empty),
            (3, 2, CellType::Empty),
            (1, 3, CellType::Empty),
            (2, 3, CellType::Empty),
            (3, 3, CellType::Empty),
        ];
        for (x, y, cell_type) in cells {
            grid.grid.set(0, y, x, cell_type);
        }
        grid
    }

    fn observation_data(agent: &Agent, position: [usize; 2]) -> Vec<f32> {
        let tensor = agent
            .get_observation_tensor(position)
            .expect("observation tensor should be constructed");
        let (_shape, data) = tensor
            .try_extract_tensor::<f32>()
            .expect("observation tensor should contain f32 values");
        data.to_vec()
    }

    #[test]
    fn tensor_layers_have_stable_channel_order() {
        assert_eq!(TensorLayer::len(), 6);
        assert_eq!(
            TensorLayer::ALL.map(|layer| layer as usize),
            [0, 1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn observation_tensor_contains_each_known_layer_at_the_expected_offset() {
        let mut agent = test_agent([3, 3], [4, 3]);
        let position = [2, 1];

        for layer in TensorLayer::ALL {
            agent
                .known_env
                .set(layer as usize, 0, 1, layer as usize as f32 + 0.5);
            agent
                .known_env
                .set(layer as usize, 1, 2, layer as usize as f32 + 1.5);
        }

        let observation = observation_data(&agent, position);
        let channel_size = 3 * 3;

        for layer in TensorLayer::ALL {
            let channel_offset = layer as usize * channel_size;
            assert_eq!(observation[channel_offset], layer as usize as f32 + 0.5);
            assert_eq!(observation[channel_offset + 4], layer as usize as f32 + 1.5);
            assert_eq!(observation[channel_offset + 1], 0.0);
        }
    }

    #[test]
    fn update_memory_maps_observation_to_memory_layers() {
        let mut agent = test_agent([3, 3], [5, 5]);
        let grid = test_grid();

        agent.update_memory([2, 2], &grid);

        let known_map = &agent.known_env;
        assert_eq!(known_map.get(TensorLayer::Finish as usize, 1, 2), &1.0);
        assert_eq!(known_map.get(TensorLayer::Visited as usize, 2, 2), &1.0);
        assert_eq!(known_map.get(TensorLayer::Wall as usize, 2, 1), &1.0);
        assert_eq!(known_map.get(TensorLayer::Traversable as usize, 1, 1), &1.0);
        assert_eq!(known_map.get(TensorLayer::Traversable as usize, 1, 3), &1.0);
        assert_eq!(known_map.get(TensorLayer::Bonus as usize, 1, 3), &1.0);
        assert_eq!(known_map.get(TensorLayer::Revealed as usize, 3, 3), &1.0);
        assert_eq!(known_map.get(TensorLayer::Wall as usize, 1, 1), &0.0);
        assert_eq!(known_map.get(TensorLayer::Finish as usize, 1, 1), &0.0);
    }

    #[test]
    fn update_memory_clips_observation_at_grid_edges() {
        let mut agent = test_agent([3, 3], [2, 2]);
        let mut grid = GridEnv::new(2, 2);
        grid.grid.set(0, 0, 0, CellType::Finish);
        grid.grid.set(0, 0, 1, CellType::Bonus);
        grid.grid.set(0, 1, 0, CellType::Wall);
        grid.grid.set(0, 1, 1, CellType::Finish);

        agent.update_memory([0, 0], &grid);

        let known_map = &agent.known_env;
        assert_eq!(known_map.get(TensorLayer::Visited as usize, 0, 0), &1.0);
        assert_eq!(known_map.get(TensorLayer::Finish as usize, 0, 0), &1.0);
        assert_eq!(known_map.get(TensorLayer::Bonus as usize, 0, 1), &1.0);
        assert_eq!(known_map.get(TensorLayer::Wall as usize, 1, 0), &1.0);
        assert_eq!(known_map.get(TensorLayer::Finish as usize, 1, 1), &1.0);
        assert_eq!(known_map.get(TensorLayer::Revealed as usize, 1, 1), &1.0);
    }

    #[test]
    fn update_memory_preserves_consumed_bonus() {
        let mut agent = test_agent([3, 3], [5, 5]);
        let grid = test_grid();
        agent.update_memory([2, 2], &grid);
        agent.known_env.set(TensorLayer::Bonus as usize, 2, 2, -1.0);

        agent.update_memory([2, 2], &grid);

        assert_eq!(
            agent.known_env.get(TensorLayer::Bonus as usize, 2, 2),
            &-1.0
        );
    }

    #[test]
    fn observation_tensor_marks_out_of_bounds_cells_as_walls() {
        let agent = test_agent([3, 3], [2, 2]);

        let observation = observation_data(&agent, [0, 0]);
        let wall_offset = TensorLayer::Wall as usize * 3 * 3;

        assert_eq!(observation.len(), TensorLayer::len() * 3 * 3);
        assert_eq!(&observation[wall_offset..wall_offset + 3], &[1.0, 1.0, 1.0]);
        assert_eq!(observation[wall_offset + 3], 1.0);
        assert_eq!(observation[wall_offset + 4], 0.0);
        assert_eq!(observation[wall_offset + 5], 0.0);
        assert_eq!(observation[wall_offset + 6], 1.0);
        assert_eq!(observation[wall_offset + 7], 0.0);
        assert_eq!(observation[wall_offset + 8], 0.0);
    }

    #[test]
    fn reset_clears_known_environment_memory() {
        let mut agent = test_agent([3, 3], [4, 3]);
        agent.known_env.set(TensorLayer::Bonus as usize, 1, 2, 1.0);
        agent.known_env.set(TensorLayer::Wall as usize, 2, 0, 1.0);

        agent.reset();

        assert!(agent.known_env.as_slice().iter().all(|&value| value == 0.0));
    }

    #[test]
    fn action_from_output_returns_error_when_values_are_empty_or_nan() {
        assert!(matches!(
            Agent::action_from_output(&[]),
            Err(AgentError::NoValidActionValues)
        ));
        assert!(matches!(
            Agent::action_from_output(&[f32::NAN, f32::NAN]),
            Err(AgentError::NoValidActionValues)
        ));
    }

    #[test]
    fn action_from_output_returns_error_when_index_is_out_of_bounds() {
        let output = vec![0.0; Action::ALL.len() + 1];
        let mut output = output;
        *output.last_mut().unwrap() = 1.0;

        assert!(matches!(
            Agent::action_from_output(&output),
            Err(AgentError::InvalidActionIndex(index)) if index == Action::ALL.len()
        ));
    }
}
