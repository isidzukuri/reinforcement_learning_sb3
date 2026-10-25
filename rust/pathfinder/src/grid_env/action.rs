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

#[cfg(test)]
mod tests {
    use super::Action;

    #[test]
    fn actions_produce_expected_deltas() {
        assert_eq!(Action::Top.delta(), [0, -1]);
        assert_eq!(Action::Bottom.delta(), [0, 1]);
        assert_eq!(Action::Left.delta(), [-1, 0]);
        assert_eq!(Action::Right.delta(), [1, 0]);
    }
}
