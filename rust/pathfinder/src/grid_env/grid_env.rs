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
