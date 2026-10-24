#[derive(Debug)]
pub struct Tensor<T> {
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<T>,
}

impl<T: Default + Clone> Tensor<T> {
    /// Creates a new 3D Tensor initialized with default values for type T.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![T::default(); channels * height * width],
        }
    }
}

impl<T> Tensor<T> {
    /// Creates a Tensor from an existing flat vector, verifying length match.
    pub fn from_vec(channels: usize, height: usize, width: usize, data: Vec<T>) -> Self {
        assert_eq!(
            channels * height * width,
            data.len(),
            "Data length does not match tensor dimensions"
        );
        Self {
            channels,
            height,
            width,
            data,
        }
    }

    /// Gets an immutable reference to a cell at (channel, row, column).
    #[inline]
    pub fn get(&self, channel: usize, row: usize, column: usize) -> &T {
        let index = channel * (self.height * self.width) + row * self.width + column;
        &self.data[index]
    }

    /// Gets a mutable reference to a cell at (channel, row, column).
    #[inline]
    pub fn get_mut(&mut self, channel: usize, row: usize, column: usize) -> &mut T {
        let index = channel * (self.height * self.width) + row * self.width + column;
        &mut self.data[index]
    }

    /// Sets the value of a cell at (channel, row, column).
    #[inline]
    pub fn set(&mut self, channel: usize, row: usize, column: usize, value: T) {
        let index = channel * (self.height * self.width) + row * self.width + column;
        self.data[index] = value;
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

    #[inline]
    pub fn channel_mut(&mut self, channel: usize) -> &mut [T] {
        let channel_size = self.height * self.width;
        let start = channel * channel_size;
        &mut self.data[start..start + channel_size]
    }

    /// Returns an immutable slice for a specific channel's contiguous data block.
    #[inline]
    pub fn channel(&self, channel: usize) -> &[T] {
        let channel_size = self.height * self.width;
        let start = channel * channel_size;
        &self.data[start..start + channel_size]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new_and_dimensions() {
        let tensor: Tensor<u8> = Tensor::new(2, 4, 5);
        assert_eq!(tensor.channels(), 2);
        assert_eq!(tensor.height(), 4);
        assert_eq!(tensor.width(), 5);
        assert_eq!(tensor.as_slice().len(), 40); // 2 * 4 * 5
        assert_eq!(*tensor.get(0, 0, 0), 0);
        assert_eq!(*tensor.get(1, 3, 4), 0);
    }

    #[test]
    fn test_tensor_get_and_set() {
        let mut tensor: Tensor<i32> = Tensor::new(3, 3, 3);

        tensor.set(0, 0, 0, 10);
        tensor.set(1, 1, 1, 20);
        tensor.set(2, 2, 2, 30);

        assert_eq!(*tensor.get(0, 0, 0), 10);
        assert_eq!(*tensor.get(1, 1, 1), 20);
        assert_eq!(*tensor.get(2, 2, 2), 30);
    }

    #[test]
    fn test_tensor_get_mut() {
        let mut tensor: Tensor<u32> = Tensor::new(1, 2, 2);
        tensor.set(0, 1, 1, 5);

        let cell = tensor.get_mut(0, 1, 1);
        *cell += 10;

        assert_eq!(*tensor.get(0, 1, 1), 15);
    }

    #[test]
    fn test_tensor_from_vec() {
        // Shape: 1 channel, 2 height, 3 width -> 6 elements total
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = Tensor::from_vec(1, 2, 3, data);

        assert_eq!(*tensor.get(0, 0, 0), 1);
        assert_eq!(*tensor.get(0, 0, 2), 3);
        assert_eq!(*tensor.get(0, 1, 0), 4);
        assert_eq!(*tensor.get(0, 1, 2), 6);
    }

    #[test]
    #[should_panic(expected = "Data length does not match tensor dimensions")]
    fn test_tensor_from_vec_mismatch_panics() {
        let data = vec![1, 2, 3];
        // 2 channels * 2 height * 2 width requires 8 elements, but only 3 provided
        let _tensor: Tensor<u8> = Tensor::from_vec(2, 2, 2, data);
    }

    #[test]
    fn test_tensor_into_vec() {
        // Create a small tensor and populate it
        let mut tensor: Tensor<i32> = Tensor::new(1, 2, 2);
        tensor.set(0, 0, 0, 10);
        tensor.set(0, 0, 1, 20);
        tensor.set(0, 1, 0, 30);
        tensor.set(0, 1, 1, 40);

        // Convert into a vector
        let data_vec = tensor.into_vec();

        // Verify the extracted vector matches the expected layout
        assert_eq!(data_vec, vec![10, 20, 30, 40]);

        // Note: Trying to use `tensor` after this point would fail
        // at compile time because `into_vec` consumes ownership (takes `self`).
    }

    #[test]
    fn test_tensor_channel_slices() {
        // Shape: 2 channels, 2 height, 3 width -> 12 elements total (6 per channel)
        let mut tensor: Tensor<i32> = Tensor::new(2, 2, 3);

        // Populate channel 0 and channel 1 with distinct values
        tensor.fill_channel(0, 10);
        tensor.fill_channel(1, 20);

        // Verify immutable channel slices
        assert_eq!(tensor.channel(0), &[10, 10, 10, 10, 10, 10]);
        assert_eq!(tensor.channel(1), &[20, 20, 20, 20, 20, 20]);

        // Verify mutable channel manipulation
        let ch0_mut = tensor.channel_mut(0);
        ch0_mut[0] = 99;
        assert_eq!(tensor.get(0, 0, 0), &99);
        // Ensure channel 1 was unaffected
        assert_eq!(tensor.get(1, 0, 0), &20);
    }

    #[test]
    fn test_tensor_fill_channel() {
        let mut tensor: Tensor<f32> = Tensor::new(3, 2, 2); // 3 channels, 4 elements each

        // Fill only channel 1 with 1.0
        tensor.fill_channel(1, 1.0);

        // Check channel 0 remains default (0.0)
        assert_eq!(tensor.channel(0), &[0.0, 0.0, 0.0, 0.0]);
        // Check channel 1 is filled with 1.0
        assert_eq!(tensor.channel(1), &[1.0, 1.0, 1.0, 1.0]);
        // Check channel 2 remains default (0.0)
        assert_eq!(tensor.channel(2), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor_fill_entire_tensor() {
        // Create a 2-channel, 2x2 tensor (8 elements total)
        let mut tensor: Tensor<i32> = Tensor::new(2, 2, 2);

        // Initially all zeros
        assert_eq!(tensor.as_slice(), &[0, 0, 0, 0, 0, 0, 0, 0]);

        // Fill the entire tensor with 42
        tensor.fill(42);

        // Verify every element across all channels, rows, and columns is 42
        assert_eq!(tensor.as_slice(), &[42, 42, 42, 42, 42, 42, 42, 42]);
        assert_eq!(*tensor.get(0, 0, 0), 42);
        assert_eq!(*tensor.get(1, 1, 1), 42);
    }
}
