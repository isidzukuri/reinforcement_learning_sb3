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

    pub const ALL: [Action; 4] = [Action::Top, Action::Bottom, Action::Left, Action::Right];
}
