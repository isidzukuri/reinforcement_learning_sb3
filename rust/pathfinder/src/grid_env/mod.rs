mod action;
mod cell_type;
mod csv_loader;
mod grid_env;
mod printer;

pub use action::Action;
pub use cell_type::CellType;
pub use csv_loader::{CsvLoader, CsvLoaderError};
pub use grid_env::GridEnv;
pub use printer::Printer;

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
        assert_eq!(CellType::Empty as u8, 0);
        assert_eq!(CellType::Finish as u8, 1);
        assert_eq!(CellType::Bonus as u8, 2);
        assert_eq!(CellType::Wall as u8, 3);

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

    #[test]
    fn csv_loader_parses_metadata_and_grid_rows() {
        let csv_path = unique_temp_file("grid_env");
        fs::write(&csv_path, "# start: 2, 3\n# path_length: 8\n0,1,0\n1,2,1\n").unwrap();

        let env = CsvLoader::call(csv_path.as_path()).unwrap();

        assert_eq!(env.width, 3);
        assert_eq!(env.height, 2);
        assert_eq!(env.start, [2, 3]);
        assert_eq!(env.path_length, 8);
        assert_eq!(
            env.grid.as_slice(),
            &[
                CellType::Empty,
                CellType::Finish,
                CellType::Empty,
                CellType::Finish,
                CellType::Bonus,
                CellType::Finish,
            ]
        );

        fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn csv_loader_rejects_empty_or_invalid_files() {
        let empty_path = unique_temp_file("empty_grid");
        fs::write(&empty_path, "").unwrap();
        assert!(matches!(
            CsvLoader::call(empty_path.as_path()),
            Err(CsvLoaderError::EmptyOrInvalidGrid)
        ));

        let invalid_metadata = unique_temp_file("invalid_grid");
        fs::write(&invalid_metadata, "# start: invalid\n").unwrap();
        assert!(matches!(
            CsvLoader::call(invalid_metadata.as_path()),
            Err(CsvLoaderError::InvalidStartMetadata)
        ));

        fs::remove_file(&empty_path).ok();
        fs::remove_file(&invalid_metadata).ok();
    }

    #[test]
    fn csv_loader_rejects_rows_with_different_widths() {
        let csv_path = unique_temp_file("ragged_grid");
        fs::write(&csv_path, "0,1\n2\n").unwrap();

        assert!(matches!(
            CsvLoader::call(csv_path.as_path()),
            Err(CsvLoaderError::InconsistentRowWidths)
        ));

        fs::remove_file(&csv_path).unwrap();
    }

    #[test]
    fn csv_loader_reports_invalid_cell_types() {
        let result = CsvLoader::from_reader(std::io::Cursor::new("0,invalid\n"));

        assert!(matches!(
            result,
            Err(CsvLoaderError::InvalidCellType(value)) if value == "invalid"
        ));
    }

    #[test]
    fn printer_renders_expected_summary_and_grid() {
        let env = GridEnv::new(2, 2);
        let path = [[0, 0], [1, 0]];

        let rendered = Printer::render(&env, Some(&path));

        assert!(rendered.starts_with("=== GridEnv Configuration ===\n"));
        assert!(rendered.contains("Size (Width, Height): (2, 2)"));
        assert!(rendered.contains("Start Coordinate: [0, 0]"));
        assert!(rendered.contains("Generated Path Length: 0"));
        assert!(rendered.contains("Actual Taken Path Length: 2"));
        assert!(rendered.contains("TYPES:"));
        assert!(rendered.contains("ACTIONS: [Top, Bottom, Left, Right]"));
        assert!(rendered.contains("=== Grid Matrix ==="));
        assert!(rendered.contains("\u{1b}[48;2;144;238;144m"));
        assert!(rendered.ends_with("============================\n"));
    }

    #[test]
    fn printer_without_path_omits_actual_taken_path_length() {
        let env = GridEnv::new(1, 1);

        let rendered = Printer::render(&env, None);

        assert!(!rendered.contains("Actual Taken Path Length"));
        assert!(rendered.contains("Generated Path Length: 0"));
        assert!(rendered.contains("=== Grid Matrix ==="));
    }
}
