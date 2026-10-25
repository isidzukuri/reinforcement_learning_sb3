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
            Self::InconsistentRowWidths => {
                f.write_str("Imported grid rows must have the same width.")
            }
            Self::InvalidStartMetadata => f.write_str("Invalid start metadata."),
            Self::InvalidPathLengthMetadata => f.write_str("Invalid path_length metadata."),
            Self::InvalidCellType(value) => write!(f, "Unknown cell type string: '{value}'"),
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
    pub fn call(path_to_file: &Path) -> Result<GridEnv, CsvLoaderError> {
        let file = File::open(path_to_file).map_err(CsvLoaderError::Io)?;
        Self::from_reader(BufReader::new(file))
    }

    pub fn from_reader(reader: impl BufRead) -> Result<GridEnv, CsvLoaderError> {
        let mut rows = Vec::new();
        let mut start = [0, 0];
        let mut path_length = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(CsvLoaderError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if Self::parse_header_line(&line, &mut start, &mut path_length)? {
                continue;
            }

            if let Some(row) = Self::parse_data_line(&line)? {
                rows.push(row);
            }
        }

        if rows.is_empty() {
            return Err(CsvLoaderError::EmptyOrInvalidGrid);
        }

        let height = rows.len();
        let width = rows[0].len();
        if rows.iter().any(|row| row.len() != width) {
            return Err(CsvLoaderError::InconsistentRowWidths);
        }
        let cells = rows.into_iter().flatten().collect();

        Ok(GridEnv {
            width,
            height,
            grid: Tensor::from_vec(1, height, width, cells),
            start,
            path_length,
        })
    }

    fn parse_header_line(
        line: &str,
        start: &mut [usize; 2],
        path_length: &mut usize,
    ) -> Result<bool, CsvLoaderError> {
        let Some(content) = line.trim().strip_prefix('#') else {
            return Ok(false);
        };

        let Some((metadata, value)) = content.split_once(':') else {
            return Ok(true);
        };

        let metadata = metadata.trim();
        let value = value.trim();

        if metadata == "start" {
            let coordinates: Result<Vec<usize>, _> = value
                .split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect();

            let coords = coordinates.map_err(|_| CsvLoaderError::InvalidStartMetadata)?;
            if coords.len() != 2 {
                return Err(CsvLoaderError::InvalidStartMetadata);
            }
            *start = [coords[0], coords[1]];
        } else if metadata == "path_length" {
            *path_length = value
                .parse::<usize>()
                .map_err(|_| CsvLoaderError::InvalidPathLengthMetadata)?;
        }

        Ok(true)
    }

    fn parse_data_line(line: &str) -> Result<Option<Vec<CellType>>, CsvLoaderError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        if trimmed.starts_with('#') {
            return Ok(None);
        }

        let row = trimmed
            .split(',')
            .map(|value| {
                let value = value.trim();
                value
                    .parse::<CellType>()
                    .map_err(|_| CsvLoaderError::InvalidCellType(value.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(if row.is_empty() { None } else { Some(row) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io::{Cursor, Read};

    #[test]
    fn from_reader_ignores_comments_and_blank_lines_and_trims_cells() {
        let grid = CsvLoader::from_reader(Cursor::new(
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
    fn from_reader_rejects_invalid_path_length_metadata() {
        assert!(matches!(
            CsvLoader::from_reader(Cursor::new("# path_length: invalid\n0\n")),
            Err(CsvLoaderError::InvalidPathLengthMetadata)
        ));
    }

    #[test]
    fn from_reader_rejects_start_metadata_with_wrong_coordinate_count() {
        assert!(matches!(
            CsvLoader::from_reader(Cursor::new("# start: 1, 2, 3\n0\n")),
            Err(CsvLoaderError::InvalidStartMetadata)
        ));
    }

    #[test]
    fn from_reader_rejects_files_without_grid_rows() {
        assert!(matches!(
            CsvLoader::from_reader(Cursor::new("# comment\n\n")),
            Err(CsvLoaderError::EmptyOrInvalidGrid)
        ));
    }

    #[test]
    fn parse_data_line_ignores_empty_and_comment_lines() {
        assert!(CsvLoader::parse_data_line("  ").unwrap().is_none());
        assert!(CsvLoader::parse_data_line(" # comment ").unwrap().is_none());
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
    fn from_reader_reports_read_errors() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }

        assert!(matches!(
            CsvLoader::from_reader(BufReader::new(FailingReader)),
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
