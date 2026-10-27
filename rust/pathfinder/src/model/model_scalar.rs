use std::fmt::Debug;
use num_traits;

pub trait ModelScalar:
    Copy
    + Debug
    + Default
    + PartialOrd
    + num_traits::Signed
    + for<'a> TryFrom<&'a ort::value::DynValue, Error = ort::Error>
{
}

impl<T> ModelScalar for T where
    T: Copy
        + Debug
        + Default
        + PartialOrd
        + num_traits::Signed
        + for<'a> TryFrom<&'a ort::value::DynValue, Error = ort::Error>
{
}