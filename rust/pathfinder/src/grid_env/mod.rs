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
    fn loaded_grid_renders_through_public_grid_environment_api() {
        let csv_path = unique_temp_file("grid_env");
        fs::write(&csv_path, "# start: 2, 3\n# path_length: 8\n0,1,0\n1,2,1\n").unwrap();

        let env = CsvLoader::call(csv_path.as_path()).unwrap();
        let rendered = Printer::render(&env, None);

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
        assert!(rendered.contains("Generated Path Length: 8"));
        assert!(rendered.contains(" 0  1  0"));
        assert!(rendered.contains(" 1  2  1"));
        fs::remove_file(&csv_path).unwrap();
    }
}
