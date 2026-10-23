use std::error::Error;
use std::path::Path;

use ort::session::Session;

pub fn call(file_path: impl AsRef<Path>) -> Result<Session, Box<dyn Error>> {
    Ok(Session::builder()?.commit_from_file(file_path.as_ref())?)
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

        assert!(result.is_err(), "expected a missing ONNX model to fail");
    }

    #[test]
    fn call_returns_error_for_invalid_onnx_payload() {
        let path = unique_temp_path("invalid_model");
        fs::write(&path, b"not an onnx model").unwrap();

        let result = call(&path);

        assert!(result.is_err(), "expected an invalid ONNX payload to fail");
        fs::remove_file(&path).ok();
    }
}
