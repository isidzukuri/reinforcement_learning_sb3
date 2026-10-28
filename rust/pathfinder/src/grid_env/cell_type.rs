use std::str::FromStr;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum CellType {
    #[default]
    Empty = 0,
    Finish = 1,
    Bonus = 2,
    Wall = 3,
}

impl FromStr for CellType {
    type Err = String;

    fn from_str(st: &str) -> Result<Self, Self::Err> {
        match st.trim().to_lowercase().as_str() {
            "0" => Ok(CellType::Empty),
            "1" => Ok(CellType::Finish),
            "2" => Ok(CellType::Bonus),
            "3" => Ok(CellType::Wall),
            other => Err(format!("Unknown cell type string: '{}'", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CellType;

    #[test]
    fn cell_types_have_expected_ids_and_counts() {
        assert_eq!(CellType::Empty as u8, 0);
        assert_eq!(CellType::Finish as u8, 1);
        assert_eq!(CellType::Bonus as u8, 2);
        assert_eq!(CellType::Wall as u8, 3);
    }

    #[test]
    fn cell_types_parsed_from_string() {
        assert_eq!("0".parse::<CellType>().unwrap(), CellType::Empty);
        assert_eq!("1".parse::<CellType>().unwrap(), CellType::Finish);
        assert_eq!("2".parse::<CellType>().unwrap(), CellType::Bonus);
        assert_eq!("3".parse::<CellType>().unwrap(), CellType::Wall);
    }
}
