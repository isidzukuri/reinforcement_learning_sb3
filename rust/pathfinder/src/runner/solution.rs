#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Solution {
    pub success: bool,
    pub timesteps: usize,
    pub path: Vec<[usize; 2]>,
}
