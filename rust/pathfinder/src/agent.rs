// use crate::model_loader::call;
use ort::session::Session as Model;
use ort::value::Tensor as OrtTensor;

use crate::tensor::Tensor;

use crate::grid_env::{Action, CellType, GridEnv};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum TensorLayer {
    Finish = 0,      // Finish Line (Detected finish coordinates)
    Visited = 1,     // Visited Memory (Historical trail of agent positions)
    Wall = 2,        // Confirmed Walls (Detected obstacle locations)
    Traversable = 3, // Confirmed Open Space (Verified safe floor/empty tiles)
    Bonus = 4,       // Confirmed Bonus (Active, uncollected bonus tiles)
    Revealed = 5,    // Explored Space (1 = revealed/explored, 0 = unexplored / fog of war)
}

impl TensorLayer {
    pub const fn len() -> usize {
        Self::ALL.len()
    }
    pub const ALL: [TensorLayer; 6] = [
        Self::Finish,
        Self::Visited,
        Self::Wall,
        Self::Traversable,
        Self::Bonus,
        Self::Revealed,
    ];
}

pub struct Agent {
    model: Model,
    window_size: [usize; 2],
    memory_grid_size: [usize; 2],
    known_env_map: Tensor<f32>,
    observation_tensor_size: usize,
}

impl Agent {
    pub fn new(model: Model, window_size: [usize; 2], memory_grid_size: [usize; 2]) -> Self {
        let observation_tensor_size = TensorLayer::len() * window_size[1] * window_size[0];
        let known_env_map =
            Tensor::new(TensorLayer::len(), memory_grid_size[1], memory_grid_size[0]);

        Self {
            model,
            window_size,
            memory_grid_size,
            known_env_map: known_env_map,
            observation_tensor_size: observation_tensor_size,
        }
    }

    pub fn reset(&mut self) {
        self.known_env_map = Tensor::new(
            TensorLayer::len(),
            self.memory_grid_size[1],
            self.memory_grid_size[0],
        );
    }

    /// Extracts the local observation window tensor around the agent's position.
    pub fn get_observation_tensor(
        &self,
        agent_pos: [usize; 2],
    ) -> Result<OrtTensor<f32>, ort::Error> {
        let agent_x = agent_pos[0] as isize;
        let agent_y = agent_pos[1] as isize;

        let window_width = self.window_size[0];
        let window_height = self.window_size[1];
        let half_window_width = (window_width as isize) / 2;
        let half_window_height = (window_height as isize) / 2;

        let memory_width = self.memory_grid_size[0] as isize;
        let memory_height = self.memory_grid_size[1] as isize;

        let num_channels = TensorLayer::ALL.len();
        let mut tensor_data = Tensor::new(num_channels, window_height, window_width);
        tensor_data.fill_channel(TensorLayer::Wall as usize, 1.0);

        let start_grid_x = agent_x - half_window_width;
        let end_grid_x = start_grid_x + (window_width as isize) - 1;

        for channel_idx in 0..num_channels {
            for dy_local in 0..window_height {
                let grid_y = agent_y + (dy_local as isize) - half_window_height;

                if !(grid_y >= 0 && grid_y < memory_height) {
                    continue;
                }

                let gy = grid_y as usize;

                let clip_start_x = start_grid_x.max(0);
                let clip_end_x = end_grid_x.min(memory_width - 1);

                if clip_start_x <= clip_end_x {
                    let length = (clip_end_x - clip_start_x + 1) as usize;
                    let dx_local_start = (clip_start_x - start_grid_x) as usize;
                    let gx_start = clip_start_x as usize;

                    // Source slice from global memory map
                    let src_idx = channel_idx
                        * (self.memory_grid_size[1] * self.memory_grid_size[0])
                        + gy * self.memory_grid_size[0]
                        + gx_start;
                    let src_slice = &self.known_env_map.as_slice()[src_idx..src_idx + length];

                    // Destination slice within the target channel view
                    let dst_start = dy_local * window_width + dx_local_start;
                    let dst_slice =
                        &mut tensor_data.channel_mut(channel_idx)[dst_start..dst_start + length];

                    dst_slice.copy_from_slice(src_slice);
                }
            }
        }

        let shape = [1, self.observation_tensor_size];
        let tensor = OrtTensor::from_array((shape, tensor_data.into_vec()))?;

        Ok(tensor)
    }

    pub fn select_action(
        &mut self,
        agent_pos: [usize; 2],
    ) -> Result<Action, Box<dyn std::error::Error>> {
        let obs_tensor = self.get_observation_tensor(agent_pos)?;
        let outputs = self.model.run(ort::inputs![obs_tensor])?;
        // let outputs = self.model.run(ort::inputs!["input" => obs_tensor]?)?;

        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let action_index = data
            .first()
            .copied()
            .expect("Model output tensor is empty; expected at least one action index")
            as usize;

        let action = Action::ALL
            .get(action_index)
            .copied()
            .expect("Model output action index out of bounds");

        Ok(action)
    }

    /// Updates the global memory map directly from the GridEnv.
    pub fn update_memory(
        &mut self,
        agent_pos: [usize; 2],
        grid_env: &GridEnv,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let window_width = self.window_size[0];
        let window_height = self.window_size[1];

        let agent_x = agent_pos[0] as isize;
        let agent_y = agent_pos[1] as isize;

        let memory_width = self.memory_grid_size[0] as isize;
        let memory_height = self.memory_grid_size[1] as isize;

        // Mark current agent position as visited in the Visited layer
        if agent_x >= 0 && agent_x < memory_width && agent_y >= 0 && agent_y < memory_height {
            self.known_env_map.set(
                TensorLayer::Visited as usize,
                agent_y as usize,
                agent_x as usize,
                1.0,
            );
        }

        grid_env.for_each_in_window(
            agent_pos,
            self.window_size,
            |local_x, local_y, grid_x, grid_y, cell_type| {
                if grid_x < 0 || grid_x >= memory_width || grid_y < 0 || grid_y >= memory_height {
                    return;
                }
                let gx = grid_x as usize;
                let gy = grid_y as usize;

                // Update Finish layer
                if cell_type == CellType::Finish {
                    self.known_env_map
                        .set(TensorLayer::Finish as usize, gy, gx, 1.0);
                }

                // Update Wall layer
                if cell_type == CellType::Wall {
                    self.known_env_map
                        .set(TensorLayer::Wall as usize, gy, gx, 1.0);
                }

                // Update Traversable layer
                if cell_type == CellType::Empty || cell_type == CellType::Bonus {
                    self.known_env_map
                        .set(TensorLayer::Traversable as usize, gy, gx, 1.0);
                }

                // Update Bonus layer (respecting special -1 lock)
                let current_bonus = *self.known_env_map.get(TensorLayer::Bonus as usize, gy, gx);
                if current_bonus != -1.0 {
                    if cell_type == CellType::Bonus {
                        self.known_env_map
                            .set(TensorLayer::Bonus as usize, gy, gx, 1.0);
                    } else {
                        self.known_env_map
                            .set(TensorLayer::Bonus as usize, gy, gx, 0.0);
                    }
                }

                // Update Revealed layer (Fog of War)
                self.known_env_map
                    .set(TensorLayer::Revealed as usize, gy, gx, 1.0);
            },
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::call;
    use std::path::Path;

    fn test_agent(window_size: [usize; 2], memory_grid_size: [usize; 2]) -> Agent {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/runtime/model.onnx");
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
                .known_env_map
                .set(layer as usize, 0, 1, layer as usize as f32 + 0.5);
            agent
                .known_env_map
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

        agent.update_memory([2, 2], &grid).unwrap();

        let known_map = &agent.known_env_map;
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

        agent.update_memory([0, 0], &grid).unwrap();

        let known_map = &agent.known_env_map;
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
        agent.update_memory([2, 2], &grid).unwrap();
        agent
            .known_env_map
            .set(TensorLayer::Bonus as usize, 2, 2, -1.0);

        agent.update_memory([2, 2], &grid).unwrap();

        assert_eq!(
            agent.known_env_map.get(TensorLayer::Bonus as usize, 2, 2),
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
        agent
            .known_env_map
            .set(TensorLayer::Bonus as usize, 1, 2, 1.0);
        agent
            .known_env_map
            .set(TensorLayer::Wall as usize, 2, 0, 1.0);

        agent.reset();

        assert!(
            agent
                .known_env_map
                .as_slice()
                .iter()
                .all(|&value| value == 0.0)
        );
    }
}

// fn invalid_input(message: impl Into<String>) -> io::Error {
//     io::Error::new(io::ErrorKind::InvalidInput, message.into())
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     fn test_agent() -> Agent {
//         let type_ids = [
//             ("empty".to_string(), 0),
//             ("finish".to_string(), 1),
//             ("bonus".to_string(), 2),
//             ("wall".to_string(), 3),
//         ]
//         .into_iter()
//         .collect();
//         Agent::new(type_ids, [3, 3], [5, 5]).unwrap()
//     }

//     #[test]
//     fn update_memory_tracks_observation_layers() {
//         let mut agent = test_agent();
//         agent
//             .update_memory([2, 2], &[vec![0, 3, 0], vec![2, 1, 0], vec![0, 0, 0]])
//             .unwrap();

//         assert_eq!(agent.known_env_map[TENSOR_LAYERS[1].1][2][2], 1);
//         assert_eq!(agent.known_env_map[TENSOR_LAYERS[0].1][2][2], 1);
//         assert_eq!(agent.known_env_map[TENSOR_LAYERS[2].1][1][2], 1);
//         assert_eq!(agent.known_env_map[TENSOR_LAYERS[4].1][2][1], 1);
//         assert_eq!(agent.known_env_map[TENSOR_LAYERS[5].1][1][2], 1);

//         assert!(agent.consume_bonus([1, 2]).unwrap());
//         assert!(!agent.consume_bonus([1, 2]).unwrap());
//     }

//     #[test]
//     fn reset_clears_memory() {
//         let mut agent = test_agent();
//         let local_state = [vec![0; 3], vec![0; 3], vec![0; 3]];
//         agent.update_memory([2, 2], &local_state).unwrap();
//         agent.reset();

//         assert!(
//             agent
//                 .known_env_map
//                 .iter()
//                 .flatten()
//                 .flatten()
//                 .all(|&cell| cell == 0)
//         );
//     }

//     #[test]
//     fn loads_and_runs_the_repository_onnx_model() {
//         let type_ids = [
//             ("empty".to_string(), 0),
//             ("finish".to_string(), 1),
//             ("bonus".to_string(), 2),
//             ("wall".to_string(), 3),
//         ]
//         .into_iter()
//         .collect();
//         let mut agent = Agent::new(type_ids, [5, 5], [15, 15]).unwrap();
//         let model_path =
//             Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/runtime/model.onnx");
//         agent.load_model(model_path).unwrap();
//         let actions = ["top", "bottom", "left", "right"].map(str::to_string);

//         assert!(actions.contains(&agent.select_action([7, 7], &actions).unwrap()));
//     }
// }
