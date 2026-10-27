use std::fmt;
use std::io;

#[derive(Debug)]
pub enum ModelLoaderError {
    SessionBuilder(ort::Error),
    LoadModel(ort::Error),
    Io(io::Error),
}

impl fmt::Display for ModelLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadModel(error) => write!(f, "failed to load ONNX model: {error}"),
            Self::Io(error) => write!(f, "failed to read ONNX file: {error}"),
            Self::SessionBuilder(error) => {
                write!(f, "failed to create ONNX session builder: {error}")
            }
        }
    }
}

impl std::error::Error for ModelLoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SessionBuilder(error) => Some(error),
            Self::LoadModel(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn test_session_builder_error_display_and_source() {
        let ort_err = ort::Error::new("failed to initialize environment");
        let error = ModelLoaderError::SessionBuilder(ort_err);

        assert_eq!(
            error.to_string(),
            "failed to create ONNX session builder: failed to initialize environment"
        );

        let source = error.source().expect("expected a source error");
        assert_eq!(source.to_string(), "failed to initialize environment");
    }

    #[test]
    fn test_load_model_error_display_and_source() {
        let ort_err = ort::Error::new("corrupted model weights");
        let error = ModelLoaderError::LoadModel(ort_err);

        assert_eq!(
            error.to_string(),
            "failed to load ONNX model: corrupted model weights"
        );

        let source = error.source().expect("expected a source error");
        assert_eq!(source.to_string(), "corrupted model weights");
    }

    #[test]
    fn test_io_error_display_and_source() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let error = ModelLoaderError::Io(io_err);

        assert_eq!(
            error.to_string(),
            "failed to read ONNX file: file not found"
        );

        let source = error.source().expect("expected a source error");
        assert_eq!(source.to_string(), "file not found");
    }
}
