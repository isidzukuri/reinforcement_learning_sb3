use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{CellType, GridEnv};
use crate::tensor::Tensor;

pub struct CsvLoader;

impl CsvLoader {
    pub fn call(path_to_file: &Path) -> Result<GridEnv, Box<dyn std::error::Error>> {
        let file = File::open(path_to_file)?;
        let reader = BufReader::new(file);

        let mut rows = Vec::new();
        let mut start = [0, 0];
        let mut path_length = 0;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Ok(true) = Self::parse_header_line(&line, &mut start, &mut path_length) {
                continue;
            }

            if let Some(row) = Self::parse_data_line(&line)? {
                rows.push(row);
            }
        }

        if rows.is_empty() {
            return Err("Imported file is empty or invalid.".into());
        }

        let height = rows.len();
        let width = rows[0].len();
        if rows.iter().any(|row| row.len() != width) {
            return Err("Imported grid rows must have the same width.".into());
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
    ) -> Result<bool, Box<dyn std::error::Error>> {
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

            let coords = coordinates.map_err(|_| "Invalid start metadata.")?;
            if coords.len() != 2 {
                return Err("Start metadata must contain two coordinates.".into());
            }
            *start = [coords[0], coords[1]];
        } else if metadata == "path_length" {
            *path_length = value
                .parse::<usize>()
                .map_err(|_| "Invalid path_length metadata.")?;
        }

        Ok(true)
    }

    fn parse_data_line(line: &str) -> Result<Option<Vec<CellType>>, Box<dyn std::error::Error>> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        if trimmed.starts_with('#') {
            return Ok(None);
        }

        let row: Result<Vec<CellType>, _> = trimmed
            .split(',')
            .map(|val| val.trim().parse::<CellType>())
            .collect();
        let row = row?;

        Ok(if row.is_empty() { None } else { Some(row) })
    }
}
