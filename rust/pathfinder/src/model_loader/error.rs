use std::fmt;

#[derive(Debug)]
pub enum ModelLoaderError {
    SessionBuilder(ort::Error),
    LoadModel(ort::Error),
}

impl fmt::Display for ModelLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionBuilder(error) => {
                write!(f, "failed to create ONNX session builder: {error}")
            }
            Self::LoadModel(error) => write!(f, "failed to load ONNX model: {error}"),
        }
    }
}

impl std::error::Error for ModelLoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SessionBuilder(error) | Self::LoadModel(error) => Some(error),
        }
    }
}
