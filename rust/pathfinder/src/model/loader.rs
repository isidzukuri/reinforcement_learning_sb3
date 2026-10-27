pub use super::error::ModelLoaderError;

use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use std::path::Path;

pub struct Loader;

impl Loader {
    pub fn call(
        file_path: impl AsRef<Path>,
    ) -> Result<(TensorElementType, Session), ModelLoaderError> {
        let builder = Session::builder().map_err(ModelLoaderError::SessionBuilder)?;
        let session = builder
            .commit_from_file(file_path.as_ref())
            .map_err(ModelLoaderError::LoadModel)?;

        let input = session
            .inputs
            .first()
            .ok_or_else(|| ModelLoaderError::LoadModel(ort::Error::new("model has no inputs")))?;
        let dtype = match &input.input_type {
            ValueType::Tensor { ty, .. } => *ty,
            _ => {
                return Err(ModelLoaderError::LoadModel(ort::Error::new(
                    "model input is not a tensor",
                )));
            }
        };

        Ok((dtype, session))
    }
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

        let result = Loader::call(&path);

        assert!(
            matches!(result, Err(ModelLoaderError::LoadModel(_))),
            "expected a missing ONNX model to fail while loading"
        );
    }

    #[test]
    fn call_returns_error_for_invalid_onnx_payload() {
        let path = unique_temp_path("invalid_model");
        fs::write(&path, b"not an onnx model").unwrap();

        let result = Loader::call(&path);

        assert!(
            matches!(result, Err(ModelLoaderError::LoadModel(_))),
            "expected an invalid ONNX payload to fail while loading"
        );
        fs::remove_file(&path).ok();
    }
}
