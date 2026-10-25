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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn model_loader_errors_report_display_and_source() {
        let error = ModelLoaderError::SessionBuilder(ort::Error::new("session builder failed"));
        assert_eq!(
            error.to_string(),
            "failed to create ONNX session builder: session builder failed"
        );
        assert!(error.source().is_some());

        let error = ModelLoaderError::LoadModel(ort::Error::new("load model failed"));
        assert_eq!(error.to_string(), "failed to load ONNX model: load model failed");
        assert!(error.source().is_some());
    }
}
