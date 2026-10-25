use super::CellType;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct GridEnv {
    pub width: usize,
    pub height: usize,
    pub grid: Tensor<CellType>,
    pub start: [usize; 2],
    pub path_length: usize,
}

impl GridEnv {
    const TENSOR_LAYER: usize = 0;

    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            grid: Tensor::new(1, height, width),
            start: [0, 0],
            path_length: 0,
        }
    }

    pub fn for_each_in_window<F>(&self, coords: [usize; 2], window: [usize; 2], mut fnc: F)
    where
        F: FnMut(usize, usize, isize, isize, CellType),
    {
        let center_x = coords[0] as isize;
        let center_y = coords[1] as isize;
        let window_width = window[0] as isize;
        let window_height = window[1] as isize;

        let half_width = window_width / 2;
        let half_height = window_height / 2;

        for local_y in 0..window[1] {
            for local_x in 0..window[0] {
                let neighbor_x = center_x + local_x as isize - half_width;
                let neighbor_y = center_y + local_y as isize - half_height;

                let cell_type = if neighbor_x >= 0
                    && neighbor_x < self.width as isize
                    && neighbor_y >= 0
                    && neighbor_y < self.height as isize
                {
                    *self
                        .grid
                        .get(Self::TENSOR_LAYER, neighbor_y as usize, neighbor_x as usize)
                } else {
                    CellType::Wall
                };

                fnc(local_x, local_y, neighbor_x, neighbor_y, cell_type);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_grid_with_default_start() {
        let env = GridEnv::new(3, 2);

        assert_eq!(env.width, 3);
        assert_eq!(env.height, 2);
        assert_eq!(env.start, [0, 0]);
        assert_eq!(env.path_length, 0);
        assert_eq!(env.grid.channels(), 1);
        assert_eq!(env.grid.width(), 3);
        assert_eq!(env.grid.height(), 2);
        assert!(
            env.grid
                .as_slice()
                .iter()
                .all(|&cell| cell == CellType::Empty)
        );
    }

    #[test]
    fn for_each_in_window_visits_cells_in_row_major_order_with_coordinates() {
        let mut env = GridEnv::new(2, 2);
        env.grid.set(0, 0, 0, CellType::Finish);
        env.grid.set(0, 0, 1, CellType::Bonus);
        env.grid.set(0, 1, 0, CellType::Wall);
        env.grid.set(0, 1, 1, CellType::Empty);
        let mut visited = Vec::new();

        env.for_each_in_window([1, 1], [2, 2], |local_x, local_y, x, y, cell_type| {
            visited.push((local_x, local_y, x, y, cell_type));
        });

        assert_eq!(
            visited,
            vec![
                (0, 0, 0, 0, CellType::Finish),
                (1, 0, 1, 0, CellType::Bonus),
                (0, 1, 0, 1, CellType::Wall),
                (1, 1, 1, 1, CellType::Empty),
            ]
        );
    }

    #[test]
    fn for_each_in_window_reports_wall_for_out_of_bounds_coordinates() {
        let mut env = GridEnv::new(2, 2);
        env.grid.set(0, 0, 0, CellType::Finish);
        env.grid.set(0, 0, 1, CellType::Bonus);
        env.grid.set(0, 1, 0, CellType::Wall);
        env.grid.set(0, 1, 1, CellType::Empty);
        let mut visited = Vec::new();

        env.for_each_in_window([0, 0], [3, 3], |local_x, local_y, x, y, cell_type| {
            visited.push((local_x, local_y, x, y, cell_type));
        });

        assert_eq!(
            visited,
            vec![
                (0, 0, -1, -1, CellType::Wall),
                (1, 0, 0, -1, CellType::Wall),
                (2, 0, 1, -1, CellType::Wall),
                (0, 1, -1, 0, CellType::Wall),
                (1, 1, 0, 0, CellType::Finish),
                (2, 1, 1, 0, CellType::Bonus),
                (0, 2, -1, 1, CellType::Wall),
                (1, 2, 0, 1, CellType::Wall),
                (2, 2, 1, 1, CellType::Empty),
            ]
        );
    }
}
