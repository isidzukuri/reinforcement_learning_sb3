use std::str::FromStr;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CellType {
    #[default]
    Empty = 0,
    Finish = 1,
    Bonus = 2,
    Wall = 3,
}

impl CellType {
    pub const fn count(self) -> Option<usize> {
        match self {
            CellType::Empty => None,
            CellType::Finish => Some(1),
            CellType::Bonus => Some(5),
            CellType::Wall => Some(15),
        }
    }
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
