use num_traits::Signed;
use ort::tensor::PrimitiveTensorElementType;
use std::fmt::Debug;

pub trait ModelScalar:
    Copy + Debug + Default + PartialOrd + Signed + PrimitiveTensorElementType + 'static
{
}

impl<T> ModelScalar for T where
    T: Copy + Debug + Default + PartialOrd + Signed + PrimitiveTensorElementType + 'static
{
}
