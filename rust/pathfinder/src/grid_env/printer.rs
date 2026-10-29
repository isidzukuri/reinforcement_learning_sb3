use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;

use super::{CellType, GridEnv};

pub struct Printer<'a> {
    env: &'a GridEnv,
    path: Option<&'a [[usize; 2]]>,
}

impl Printer<'_> {
    const BASE_HEADER_CAPACITY: usize = 256;
    const BYTES_PER_CELL_ESTIMATE: usize = 15;

    pub fn call(env: &GridEnv, path: Option<&[[usize; 2]]>) {
        let printer = Printer { env, path };

        print!("{}", printer.render_to_string());
    }

    fn render_to_string(&self) -> String {
        let mut output = String::with_capacity(self.capacity());

        output.push_str("=== GridEnv Configuration ===\n");
        self.output_size(&mut output);
        self.output_path(&mut output);
        self.output_cell_types(&mut output);
        output.push_str("ACTIONS: [Top, Bottom, Left, Right]\n");
        self.output_matrix(&mut output);
        output
    }

    fn capacity(&self) -> usize {
        Self::BASE_HEADER_CAPACITY
            + (self.env.width * self.env.height * Self::BYTES_PER_CELL_ESTIMATE)
    }

    fn output_size(&self, output: &mut String) {
        let _ = writeln!(
            output,
            "Size (Width, Height): ({}, {})",
            self.env.width, self.env.height
        );
    }

    fn output_path(&self, output: &mut String) {
        let _ = writeln!(output, "Start Coordinate: {:?}", self.env.start);
        let _ = writeln!(output, "Generated Path Length: {}", self.env.path_length);

        if let Some(path) = self.path {
            let _ = writeln!(output, "Actual Taken Path Length: {}", path.len());
        }
    }

    fn output_cell_types(&self, output: &mut String) {
        let cells_count = self.calculate_cell_counts();

        output.push_str("TYPES:\n");
        for (name, count) in cells_count {
            let _ = writeln!(
                output,
                "  - {:?}: id: {}, count: {:?}",
                name, *name as u8, count
            );
        }
    }
    
    fn output_matrix(&self, output: &mut String) {
        let _ = writeln!(output, "\n=== Grid Matrix ===");

        let path_cells = self.path_cells();

        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let cell = *self.env.grid.get(0, y, x);
                let cell_id = cell as u8;

                if x > 0 {
                    output.push_str(" ");
                }

                if path_cells.contains(&(x, y)) {
                    let _ = write!(output, "\x1b[48;2;144;238;144m{:2}\x1b[0m", cell_id);
                } else {
                    let _ = write!(output, "{:2}", cell_id);
                }
            }
            output.push_str("\n");
        }
        let _ = writeln!(output, "============================");
    }

    fn calculate_cell_counts(&self) -> BTreeMap<&CellType, usize> {
        let mut counter = BTreeMap::new();
        for cell in self.env.grid.as_slice() {
            *counter.entry(cell).or_insert(0) += 1;
        }
        counter
    }

    fn path_cells(&self) -> HashSet<(usize, usize)> {
        match self.path {
            Some(path) => path.iter().map(|coord| (coord[0], coord[1])).collect(),
            None => HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_includes_expected_summary_grid_and_highlights_path() {
        let env = GridEnv::new(2, 2);
        let path = [[0, 0], [1, 0]];

        let printer = Printer {
            env: &env,
            path: Some(&path),
        };
        let rendered = printer.render_to_string();

        assert!(rendered.starts_with("=== GridEnv Configuration ===\n"));
        assert!(rendered.contains("Size (Width, Height): (2, 2)"));
        assert!(rendered.contains("Start Coordinate: [0, 0]"));
        assert!(rendered.contains("Generated Path Length: 0"));
        assert!(rendered.contains("Actual Taken Path Length: 2"));
        assert!(rendered.contains("TYPES:"));
        assert!(rendered.contains("Empty: id: 0, count: 4"));
        assert!(rendered.contains("ACTIONS: [Top, Bottom, Left, Right]"));
        assert!(rendered.contains("=== Grid Matrix ==="));
        assert!(rendered.contains("\u{1b}[48;2;144;238;144m"));
        assert!(rendered.ends_with("============================\n"));
    }

    #[test]
    fn render_without_path_omits_actual_taken_path_length() {
        let env = GridEnv::new(1, 1);

        let printer = Printer {
            env: &env,
            path: None,
        };
        let rendered = printer.render_to_string();

        assert!(!rendered.contains("Actual Taken Path Length"));
        assert!(rendered.contains("Generated Path Length: 0"));
        assert!(rendered.contains("=== Grid Matrix ==="));
    }
}
