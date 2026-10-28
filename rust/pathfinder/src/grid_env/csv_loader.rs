use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{CellType, GridEnv};
use crate::tensor::Tensor;

pub struct CsvLoader;

#[derive(Debug)]
pub enum CsvLoaderError {
    Io(io::Error),
    EmptyOrInvalidGrid,
    InconsistentRowWidths,
    InvalidStartMetadata,
    InvalidPathLengthMetadata,
    InvalidCellType(String),
}

impl fmt::Display for CsvLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to read CSV file: {error}"),
            Self::EmptyOrInvalidGrid => f.write_str("Imported file is empty or invalid."),
            Self::InvalidStartMetadata => f.write_str("Invalid start metadata."),
            Self::InvalidPathLengthMetadata => f.write_str("Invalid path_length metadata."),
            Self::InvalidCellType(value) => write!(f, "Unknown cell type string: '{value}'"),
            Self::InconsistentRowWidths => {
                f.write_str("Imported grid rows must have the same width.")
            }
        }
    }
}

impl std::error::Error for CsvLoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::EmptyOrInvalidGrid
            | Self::InconsistentRowWidths
            | Self::InvalidStartMetadata
            | Self::InvalidPathLengthMetadata
            | Self::InvalidCellType(_) => None,
        }
    }
}

impl CsvLoader {
    pub fn call(path_to_file: impl AsRef<Path>) -> Result<GridEnv, CsvLoaderError> {
        let file = File::open(path_to_file).map_err(CsvLoaderError::Io)?;
        Self::load_from(BufReader::new(file))
    }

    pub fn load_from(reader: impl BufRead) -> Result<GridEnv, CsvLoaderError> {
        let mut cells = Vec::new();
        let mut width = None;
        let mut start = [0, 0];
        let mut path_length = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(CsvLoaderError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if Self::parse_header_line(&trimmed, &mut start, &mut path_length)? {
                continue;
            }

            let Some(row) = Self::parse_data_line(trimmed)? else {
                continue;
            };

            match width {
                None => width = Some(row.len()),
                Some(expected_width) if row.len() != expected_width => {
                    return Err(CsvLoaderError::InconsistentRowWidths);
                }
                _ => {}
            }

            cells.extend(row);
        }

        let width = width.ok_or(CsvLoaderError::EmptyOrInvalidGrid)?;
        let height = cells.len() / width;

        Ok(GridEnv {
            width,
            height,
            start,
            path_length,
            grid: Tensor::from_vec(1, height, width, cells),
        })
    }

    fn parse_header_line(
        line: &str,
        start: &mut [usize; 2],
        path_length: &mut usize,
    ) -> Result<bool, CsvLoaderError> {
        let Some(content) = line.strip_prefix('#') else {
            return Ok(false);
        };

        let Some((metadata, value)) = content.split_once(':') else {
            return Ok(true);
        };

        let metadata = metadata.trim();
        let value = value.trim();

        match metadata {
            "start" => *start = Self::parse_start_metadata(value)?,
            "path_length" => *path_length = Self::parse_path_length_metadata(value)?,
            _ => {}
        }

        Ok(true)
    }

    fn parse_start_metadata(value: &str) -> Result<[usize; 2], CsvLoaderError> {
        let coords: Vec<usize> = value
            .split(',')
            .map(|st| {
                st.trim()
                    .parse()
                    .map_err(|_| CsvLoaderError::InvalidStartMetadata)
            })
            .collect::<Result<_, _>>()?;

        coords
            .try_into()
            .map_err(|_| CsvLoaderError::InvalidStartMetadata)
    }

    fn parse_path_length_metadata(value: &str) -> Result<usize, CsvLoaderError> {
        value
            .parse::<usize>()
            .map_err(|_| CsvLoaderError::InvalidPathLengthMetadata)
    }

    fn parse_data_line(line: &str) -> Result<Option<Vec<CellType>>, CsvLoaderError> {
        let row = line
            .split(',')
            .map(Self::parse_cell)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Some(row))
    }

    fn parse_cell(value: &str) -> Result<CellType, CsvLoaderError> {
        let value = value.trim();
        value
            .parse::<CellType>()
            .map_err(|_| CsvLoaderError::InvalidCellType(value.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io::{Cursor, Read};

    #[test]
    fn load_from_ignores_comments_and_blank_lines_and_trims_cells() {
        let grid = CsvLoader::load_from(Cursor::new(
            "\n# comment without metadata\n  # unknown: value\n # start: 1, 0 \n # path_length: 4 \n 0, 1 \n\n",
        ))
        .unwrap();

        assert_eq!(grid.width, 2);
        assert_eq!(grid.height, 1);
        assert_eq!(grid.start, [1, 0]);
        assert_eq!(grid.path_length, 4);
        assert_eq!(grid.grid.as_slice(), &[CellType::Empty, CellType::Finish]);
    }

    #[test]
    fn load_from_rejects_invalid_path_length_metadata() {
        assert!(matches!(
            CsvLoader::load_from(Cursor::new("# path_length: invalid\n0\n")),
            Err(CsvLoaderError::InvalidPathLengthMetadata)
        ));
    }

    #[test]
    fn load_from_rejects_start_metadata_with_wrong_coordinate_count() {
        assert!(matches!(
            CsvLoader::load_from(Cursor::new("# start: 1, 2, 3\n0\n")),
            Err(CsvLoaderError::InvalidStartMetadata)
        ));
    }

    #[test]
    fn csv_loader_reports_invalid_cell_types() {
        let result = CsvLoader::load_from(std::io::Cursor::new("0,invalid\n"));

        assert!(matches!(
            result,
            Err(CsvLoaderError::InvalidCellType(value)) if value == "invalid"
        ));
    }

    #[test]
    fn load_from_rejects_files_without_grid_rows() {
        assert!(matches!(
            CsvLoader::load_from(Cursor::new("# comment\n\n")),
            Err(CsvLoaderError::EmptyOrInvalidGrid)
        ));
    }

    #[test]
    fn load_from_rejects_rows_with_different_widths() {
        assert!(matches!(
            CsvLoader::load_from(Cursor::new("0,1\n2\n")),
            Err(CsvLoaderError::InconsistentRowWidths)
        ));
    }

    #[test]
    fn call_rejects_empty_files() {
        let path =
            std::env::temp_dir().join(format!("empty_pathfinder_grid_{}.csv", std::process::id()));
        std::fs::write(&path, "").unwrap();

        let result = CsvLoader::call(&path);
        std::fs::remove_file(&path).unwrap();

        assert!(matches!(result, Err(CsvLoaderError::EmptyOrInvalidGrid)));
    }

    #[test]
    fn call_rejects_invalid_metadata() {
        let path = std::env::temp_dir().join(format!(
            "invalid_pathfinder_grid_{}.csv",
            std::process::id()
        ));
        std::fs::write(&path, "# start: invalid\n").unwrap();

        let result = CsvLoader::call(&path);
        std::fs::remove_file(&path).unwrap();

        assert!(matches!(result, Err(CsvLoaderError::InvalidStartMetadata)));
    }

    #[test]
    fn call_reports_file_open_errors() {
        let missing_path = std::env::temp_dir().join(format!(
            "missing_pathfinder_grid_{}.csv",
            std::process::id()
        ));

        assert!(matches!(
            CsvLoader::call(&missing_path),
            Err(CsvLoaderError::Io(_))
        ));
    }

    #[test]
    fn load_from_reports_read_errors() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }

        assert!(matches!(
            CsvLoader::load_from(BufReader::new(FailingReader)),
            Err(CsvLoaderError::Io(_))
        ));
    }

    #[test]
    fn error_display_and_sources_are_consistent() {
        let io_error = CsvLoaderError::Io(io::Error::other("disk unavailable"));
        assert!(io_error.to_string().contains("failed to read CSV file"));
        assert!(io_error.source().is_some());

        let invalid_cell = CsvLoaderError::InvalidCellType("unknown".to_owned());
        assert_eq!(
            invalid_cell.to_string(),
            "Unknown cell type string: 'unknown'"
        );
        assert!(invalid_cell.source().is_none());

        assert_eq!(
            CsvLoaderError::InconsistentRowWidths.to_string(),
            "Imported grid rows must have the same width."
        );
        assert_eq!(
            CsvLoaderError::InvalidStartMetadata.to_string(),
            "Invalid start metadata."
        );
        assert_eq!(
            CsvLoaderError::InvalidPathLengthMetadata.to_string(),
            "Invalid path_length metadata."
        );
        assert_eq!(
            CsvLoaderError::EmptyOrInvalidGrid.to_string(),
            "Imported file is empty or invalid."
        );
    }
}
