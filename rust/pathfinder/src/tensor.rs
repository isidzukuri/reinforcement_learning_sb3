#[derive(Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    channels: usize,
    height: usize,
    width: usize,
    channel_size: usize,
}

impl<T: Default + Clone> Tensor<T> {
    /// Creates a new Tensor initialized with default values.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        let channel_size = height
            .checked_mul(width)
            .expect("Height and width overflow");
        let total_size = channels
            .checked_mul(channel_size)
            .expect("Tensor dimensions overflow");

        Self {
            data: vec![T::default(); total_size],
            channels,
            height,
            width,
            channel_size,
        }
    }
}

impl<T> Tensor<T> {
    /// Creates a Tensor from an existing flat vector, verifying length match.
    pub fn from_vec(channels: usize, height: usize, width: usize, data: Vec<T>) -> Self {
        let channel_size = height
            .checked_mul(width)
            .expect("Height and width overflow");
        let expected_len = channels
            .checked_mul(channel_size)
            .expect("Tensor dimensions overflow");

        assert_eq!(
            expected_len,
            data.len(),
            "Data length does not match tensor dimensions"
        );

        Self {
            data,
            channels,
            height,
            width,
            channel_size,
        }
    }

    /// Gets an immutable reference to a cell at (channel, row, column).
    #[inline]
    pub fn get(&self, channel: usize, row: usize, column: usize) -> &T {
        let idx = self.index(channel, row, column);
        &self.data[idx]
    }

    /// Gets a mutable reference to a cell at (channel, row, column).
    #[inline]
    pub fn get_mut(&mut self, channel: usize, row: usize, column: usize) -> &mut T {
        let idx = self.index(channel, row, column);
        &mut self.data[idx]
    }

    /// Sets the value of a cell at (channel, row, column).
    #[inline]
    pub fn set(&mut self, channel: usize, row: usize, column: usize, value: T) {
        let idx = self.index(channel, row, column);
        self.data[idx] = value;
    }

    /// Returns the number of channels.
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the height of the tensor.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the width of the tensor.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns an immutable slice of the underlying flat buffer.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consumes the Tensor and returns the underlying flat `Vec<T>` with zero element copying.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns a mutable slice for a specific channel's contiguous data block.
    #[inline]
    pub fn channel_mut(&mut self, channel: usize) -> &mut [T] {
        let range = self.channel_range(channel);
        &mut self.data[range]
    }

    /// Returns an immutable slice for a specific channel's contiguous data block.
    #[inline]
    pub fn channel(&self, channel: usize) -> &[T] {
        let range = self.channel_range(channel);
        &self.data[range]
    }

    /// Fills an entire channel's memory block with a specified value.
    #[inline]
    pub fn fill_channel(&mut self, channel: usize, value: T)
    where
        T: Clone,
    {
        self.channel_mut(channel).fill(value);
    }

    /// Fills the entire tensor's flat underlying buffer with a specified value.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    /// Computes the flat buffer index for a given (channel, row, column).
    #[inline]
    fn index(&self, channel: usize, row: usize, column: usize) -> usize {
        channel * self.channel_size + row * self.width + column
    }

    /// Computes the flat buffer range for a given channel.
    #[inline]
    fn channel_range(&self, channel: usize) -> std::ops::Range<usize> {
        let start = channel * self.channel_size;
        start..start + self.channel_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let tensor: Tensor<u8> = Tensor::new(2, 4, 5);
        assert_eq!(tensor.as_slice().len(), 40);
        assert_eq!(*tensor.get(0, 0, 0), 0);
        assert_eq!(*tensor.get(1, 3, 4), 0);
    }

    #[test]
    fn test_tensor_from_vec() {
        let tensor = Tensor::from_vec(1, 2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(tensor.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "Data length does not match tensor dimensions")]
    fn test_tensor_from_vec_mismatch_panics() {
        let _tensor: Tensor<u8> = Tensor::from_vec(2, 2, 2, vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "Height and width overflow")]
    fn test_tensor_new_panics_on_height_width_overflow() {
        let _tensor: Tensor<u8> = Tensor::new(1, usize::MAX, 2);
    }

    #[test]
    #[should_panic(expected = "Tensor dimensions overflow")]
    fn test_tensor_new_panics_on_channel_overflow() {
        let _tensor: Tensor<u8> = Tensor::new(usize::MAX, 2, 2);
    }

    #[test]
    #[should_panic(expected = "Height and width overflow")]
    fn test_tensor_from_vec_panics_on_height_width_overflow() {
        let _tensor: Tensor<u8> = Tensor::from_vec(1, usize::MAX, 2, vec![]);
    }

    #[test]
    #[should_panic(expected = "Tensor dimensions overflow")]
    fn test_tensor_from_vec_panics_on_channel_overflow() {
        let _tensor: Tensor<u8> = Tensor::from_vec(usize::MAX, 2, 2, vec![0]);
    }

    #[test]
    fn test_tensor_get() {
        let tensor = Tensor::from_vec(2, 2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(*tensor.get(1, 0, 1), 6);
    }

    #[test]
    fn test_tensor_get_mut() {
        let mut tensor: Tensor<u32> = Tensor::new(1, 2, 2);
        *tensor.get_mut(0, 1, 1) = 15;
        assert_eq!(*tensor.get(0, 1, 1), 15);
    }

    #[test]
    fn test_tensor_set() {
        let mut tensor: Tensor<i32> = Tensor::new(1, 2, 2);
        tensor.set(0, 1, 0, 42);
        assert_eq!(*tensor.get(0, 1, 0), 42);
    }

    #[test]
    fn test_tensor_channels() {
        let tensor: Tensor<u8> = Tensor::new(2, 4, 5);
        assert_eq!(tensor.channels(), 2);
    }

    #[test]
    fn test_tensor_height() {
        let tensor: Tensor<u8> = Tensor::new(2, 4, 5);
        assert_eq!(tensor.height(), 4);
    }

    #[test]
    fn test_tensor_width() {
        let tensor: Tensor<u8> = Tensor::new(2, 4, 5);
        assert_eq!(tensor.width(), 5);
    }

    #[test]
    fn test_tensor_as_slice() {
        let tensor = Tensor::from_vec(1, 2, 2, vec![10, 20, 30, 40]);
        assert_eq!(tensor.as_slice(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_tensor_into_vec() {
        let tensor = Tensor::from_vec(1, 2, 2, vec![10, 20, 30, 40]);
        assert_eq!(tensor.into_vec(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_tensor_channel_mut() {
        let mut tensor: Tensor<i32> = Tensor::new(2, 2, 2);
        tensor.channel_mut(1).copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(tensor.as_slice(), &[0, 0, 0, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_tensor_channel() {
        let tensor = Tensor::from_vec(2, 2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(tensor.channel(1), &[5, 6, 7, 8]);
    }

    #[test]
    fn test_tensor_fill_channel() {
        let mut tensor: Tensor<i32> = Tensor::new(2, 2, 2);
        tensor.fill_channel(1, 7);
        assert_eq!(tensor.as_slice(), &[0, 0, 0, 0, 7, 7, 7, 7]);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor: Tensor<i32> = Tensor::new(2, 2, 2);
        tensor.fill(42);
        assert_eq!(tensor.as_slice(), &[42, 42, 42, 42, 42, 42, 42, 42]);
    }
}
