use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    Empty,
    Finish,
    Bonus,
    Wall,
}

impl CellType {
    pub const fn id(self) -> u8 {
        match self {
            CellType::Empty => 0,
            CellType::Finish => 1,
            CellType::Bonus => 2,
            CellType::Wall => 3,
        }
    }

    pub const fn count(self) -> Option<usize> {
        match self {
            CellType::Empty => None,
            CellType::Finish => Some(1),
            CellType::Bonus => Some(5),
            CellType::Wall => Some(15),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Top,
    Bottom,
    Left,
    Right,
}

impl Action {
    pub const fn delta(self) -> [isize; 2] {
        match self {
            Action::Top => [0, -1],
            Action::Bottom => [0, 1],
            Action::Left => [-1, 0],
            Action::Right => [1, 0],
        }
    }

    // pub const ALL: [Action; 4] = [
    //     Action::Top,
    //     Action::Bottom,
    //     Action::Left,
    //     Action::Right,
    // ];
}

#[derive(Debug)]
pub struct GridEnv {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<u8>>,
    pub start: [usize; 2],
    pub path_length: usize,
}

impl GridEnv {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            grid: vec![vec![CellType::Empty.id(); width]; height],
            start: [0, 0],
            path_length: 0,
        }
    }

    pub fn get_state(&self, coords: [usize; 2], window: [usize; 2]) -> Vec<Vec<u8>> {
        let center_x = coords[0] as isize;
        let center_y = coords[1] as isize;
        let window_width = window[0] as isize;
        let window_height = window[1] as isize;

        let half_width = window_width / 2;
        let half_height = window_height / 2;

        let mut sub_grid = Vec::with_capacity(window[1]);

        for dy in -half_height..=half_height {
            let mut row = Vec::with_capacity(window[0]);
            for dx in -half_width..=half_width {
                let neighbor_x = center_x + dx;
                let neighbor_y = center_y + dy;

                if neighbor_x >= 0
                    && neighbor_x < self.width as isize
                    && neighbor_y >= 0
                    && neighbor_y < self.height as isize
                {
                    row.push(self.grid[neighbor_y as usize][neighbor_x as usize]);
                } else {
                    row.push(CellType::Wall.id());
                }
            }
            sub_grid.push(row);
        }

        sub_grid
    }

    pub fn from_csv(path_to_file: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path_to_file)?;
        let reader = BufReader::new(file);

        let mut new_grid = Vec::new();
        let mut start = [0, 0];
        let mut path_length = 0;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Check for metadata lines starting with '#'
            if trimmed.starts_with('#') {
                let content = &trimmed[1..]; // Remove '#'
                if let Some((metadata, value)) = content.split_once(':') {
                    let metadata = metadata.trim();
                    let value = value.trim();

                    if metadata == "start" {
                        let coordinates: Result<Vec<usize>, _> = value
                            .split(',')
                            .map(|s| s.trim().parse::<usize>())
                            .collect();

                        let coords = coordinates.map_err(|_| "Invalid start metadata.")?;
                        if coords.len() != 2 {
                            return Err("Start metadata must contain two coordinates.".into());
                        }
                        start = [coords[0], coords[1]];
                    } else if metadata == "path_length" {
                        path_length = value
                            .parse::<usize>()
                            .map_err(|_| "Invalid path_length metadata.")?;
                    }
                }
                continue;
            }

            // Parse normal grid row (comma-separated integers)
            let row: Result<Vec<u8>, _> = trimmed
                .split(',')
                .map(|val| val.trim().parse::<u8>())
                .collect();

            let row = row?;
            if !row.is_empty() {
                new_grid.push(row);
            }
        }

        if new_grid.is_empty() {
            return Err("Imported file is empty or invalid.".into());
        }

        let height = new_grid.len();
        let width = new_grid[0].len();

        Ok(Self {
            width,
            height,
            grid: new_grid,
            start,
            path_length,
        })
    }

    pub fn print_to_stdout(&self, path: Option<&[[usize; 2]]>) {
        println!("=== GridEnv Configuration ===");
        println!("Size (Width, Height): ({}, {})", self.width, self.height);
        println!("Start Coordinate: {:?}", self.start);
        println!("Generated Path Length: {}", self.path_length);

        if let Some(pt) = path {
            println!("Actual Taken Path Length: {}", pt.len());
        }

        println!("TYPES:");
        let types = [
            ("Empty", CellType::Empty.id(), CellType::Empty.count()),
            ("Finish", CellType::Finish.id(), CellType::Finish.count()),
            ("Bonus", CellType::Bonus.id(), CellType::Bonus.count()),
            ("Wall", CellType::Wall.id(), CellType::Wall.count()),
        ];
        for (name, id, count) in types {
            println!("  - {}: id: {}, count: {:?}", name, id, count);
        }

        println!("ACTIONS: [Top, Bottom, Left, Right]");
        println!("\n=== Grid Matrix ===");

        let path_cells: HashSet<(usize, usize)> = match path {
            Some(pt) => pt.iter().map(|coord| (coord[0], coord[1])).collect(),
            None => HashSet::new(),
        };

        for (y, row) in self.grid.iter().enumerate() {
            let mut cells = Vec::with_capacity(row.len());
            for (x, &cell) in row.iter().enumerate() {
                let mut formatted_cell = format!("{:2}", cell);

                if path_cells.contains(&(x, y)) {
                    // ANSI escape code for light green background: RGB(144, 238, 144)
                    formatted_cell = format!("\x1b[48;2;144;238;144m{}\x1b[0m", formatted_cell);
                }
                cells.push(formatted_cell);
            }
            println!("{}", cells.join(" "));
        }
        println!("============================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{unique}.csv"))
    }

    #[test]
    fn cell_types_have_expected_ids_and_counts() {
        assert_eq!(CellType::Empty.id(), 0);
        assert_eq!(CellType::Finish.id(), 1);
        assert_eq!(CellType::Bonus.id(), 2);
        assert_eq!(CellType::Wall.id(), 3);

        assert_eq!(CellType::Empty.count(), None);
        assert_eq!(CellType::Finish.count(), Some(1));
        assert_eq!(CellType::Bonus.count(), Some(5));
        assert_eq!(CellType::Wall.count(), Some(15));
    }

    #[test]
    fn actions_produce_expected_deltas() {
        assert_eq!(Action::Top.delta(), [0, -1]);
        assert_eq!(Action::Bottom.delta(), [0, 1]);
        assert_eq!(Action::Left.delta(), [-1, 0]);
        assert_eq!(Action::Right.delta(), [1, 0]);
    }

    #[test]
    fn new_creates_empty_grid_with_default_start() {
        let env = GridEnv::new(3, 2);

        assert_eq!(env.width, 3);
        assert_eq!(env.height, 2);
        assert_eq!(env.start, [0, 0]);
        assert_eq!(env.path_length, 0);
        assert_eq!(env.grid, vec![vec![0, 0, 0], vec![0, 0, 0]]);
    }

    #[test]
    fn get_state_returns_window_and_fills_out_of_bounds_with_walls() {
        let mut env = GridEnv::new(3, 3);
        env.grid[0][0] = CellType::Bonus.id();

        let state = env.get_state([0, 0], [3, 3]);

        assert_eq!(state.len(), 3);
        assert!(state.iter().all(|row| row.len() == 3));
        assert_eq!(state[1][1], CellType::Bonus.id());
        assert_eq!(state[0][0], CellType::Wall.id());
    }

    #[test]
    fn from_csv_parses_metadata_and_grid_rows() {
        let csv_path = unique_temp_file("grid_env");
        fs::write(&csv_path, "# start: 2, 3\n# path_length: 8\n0,1,0\n1,2,1\n").unwrap();

        let env = GridEnv::from_csv(csv_path.as_path()).unwrap();

        assert_eq!(env.width, 3);
        assert_eq!(env.height, 2);
        assert_eq!(env.start, [2, 3]);
        assert_eq!(env.path_length, 8);
        assert_eq!(env.grid, vec![vec![0, 1, 0], vec![1, 2, 1]]);

        fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn from_csv_rejects_empty_or_invalid_files() {
        let empty_path = unique_temp_file("empty_grid");
        fs::write(&empty_path, "").unwrap();
        assert!(GridEnv::from_csv(empty_path.as_path()).is_err());

        let invalid_metadata = unique_temp_file("invalid_grid");
        fs::write(&invalid_metadata, "# start: invalid\n").unwrap();
        assert!(GridEnv::from_csv(invalid_metadata.as_path()).is_err());

        fs::remove_file(&empty_path).ok();
        fs::remove_file(&invalid_metadata).ok();
    }
}
