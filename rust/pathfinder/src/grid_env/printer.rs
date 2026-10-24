use std::collections::HashSet;

use super::{CellType, GridEnv};

pub struct Printer;

impl Printer {
    pub fn render(env: &GridEnv, path: Option<&[[usize; 2]]>) -> String {
        let mut output = String::new();
        output.push_str("=== GridEnv Configuration ===\n");
        output.push_str(&format!(
            "Size (Width, Height): ({}, {})\n",
            env.width, env.height
        ));
        output.push_str(&format!("Start Coordinate: {:?}\n", env.start));
        output.push_str(&format!("Generated Path Length: {}\n", env.path_length));

        if let Some(path) = path {
            output.push_str(&format!("Actual Taken Path Length: {}\n", path.len()));
        }

        output.push_str("TYPES:\n");
        let types = [
            ("Empty", CellType::Empty as u8, CellType::Empty.count()),
            ("Finish", CellType::Finish as u8, CellType::Finish.count()),
            ("Bonus", CellType::Bonus as u8, CellType::Bonus.count()),
            ("Wall", CellType::Wall as u8, CellType::Wall.count()),
        ];
        for (name, id, count) in types {
            output.push_str(&format!("  - {}: id: {}, count: {:?}\n", name, id, count));
        }
        output.push_str("ACTIONS: [Top, Bottom, Left, Right]\n");
        output.push_str("\n=== Grid Matrix ===\n");

        let path_cells: HashSet<(usize, usize)> = match path {
            Some(path) => path.iter().map(|coord| (coord[0], coord[1])).collect(),
            None => HashSet::new(),
        };

        for y in 0..env.height {
            let mut cells = Vec::with_capacity(env.width);
            for x in 0..env.width {
                let cell = *env.grid.get(0, y, x);
                let mut formatted_cell = format!("{:2}", cell as u8);

                if path_cells.contains(&(x, y)) {
                    formatted_cell = format!("\x1b[48;2;144;238;144m{}\x1b[0m", formatted_cell);
                }
                cells.push(formatted_cell);
            }
            output.push_str(&cells.join(" "));
            output.push('\n');
        }
        output.push_str("============================\n");
        output
    }

    pub fn print(env: &GridEnv, path: Option<&[[usize; 2]]>) {
        print!("{}", Self::render(env, path));
    }
}
