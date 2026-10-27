mod error;

pub use error::ModelLoaderError;

use ort::session::Session;
use std::path::Path;

pub fn call(file_path: impl AsRef<Path>) -> Result<Session, ModelLoaderError> {
    let model_bytes = std::fs::read(file_path).map_err(ModelLoaderError::Io)?;
    from_bytes(&model_bytes)
}

pub fn from_bytes(model_bytes: &[u8]) -> Result<Session, ModelLoaderError> {
    let builder = Session::builder().map_err(ModelLoaderError::SessionBuilder)?;
    builder
        .commit_from_memory(model_bytes)
        .map_err(ModelLoaderError::LoadModel)
}

pub fn from_bytes_with_intra_threads(
    model_bytes: &[u8],
    num_threads: usize,
) -> Result<Session, ModelLoaderError> {
    let builder = Session::builder()
        .and_then(|builder| builder.with_intra_threads(num_threads))
        .map_err(ModelLoaderError::SessionBuilder)?;
    builder
        .commit_from_memory(model_bytes)
        .map_err(ModelLoaderError::LoadModel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{unique}.onnx"))
    }

    #[test]
    fn call_returns_error_for_missing_file() {
        let path = unique_temp_path("missing_model");

        let result = call(&path);

        assert!(
            matches!(result, Err(ModelLoaderError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound),
            "expected a missing ONNX model to fail while reading the file"
        );
    }

    #[test]
    fn call_returns_error_for_invalid_onnx_payload() {
        let path = unique_temp_path("invalid_model");
        fs::write(&path, b"not an onnx model").unwrap();

        let result = call(&path);

        assert!(
            matches!(result, Err(ModelLoaderError::LoadModel(_))),
            "expected an invalid ONNX payload to fail while loading"
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn from_bytes_returns_error_for_invalid_onnx_payload() {
        let result = from_bytes(b"not an onnx model");

        assert!(
            matches!(result, Err(ModelLoaderError::LoadModel(_))),
            "expected an invalid in-memory ONNX payload to fail while loading"
        );
    }
}
