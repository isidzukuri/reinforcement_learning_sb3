#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayer {
    Finish = 0,      // Finish Line (Detected finish coordinates)
    Visited = 1,     // Visited Memory (Historical trail of agent positions)
    Wall = 2,        // Confirmed Walls (Detected obstacle locations)
    Traversable = 3, // Confirmed Open Space (Verified safe floor/empty tiles)
    Bonus = 4,       // Confirmed Bonus (Active, uncollected bonus tiles)
    Revealed = 5,    // Explored Space (1 = revealed/explored, 0 = unexplored / fog of war)
}

impl TensorLayer {
    pub const fn len() -> usize {
        Self::ALL.len()
    }

    pub const ALL: [TensorLayer; 6] = [
        Self::Finish,
        Self::Visited,
        Self::Wall,
        Self::Traversable,
        Self::Bonus,
        Self::Revealed,
    ];
}
