mod agent;
mod error;
mod tensor_layer;

pub use agent::{Agent, DynAgent};
pub use error::AgentError;
pub use tensor_layer::TensorLayer;

use ort::session::Session as Model;
use ort::tensor::TensorElementType;

pub fn new_by_type(
    data_type: TensorElementType,
    model: Model,
    window_size: [usize; 2],
    memory_dims: [usize; 2],
) -> Result<DynAgent, AgentError> {
    let agent = match data_type {
        TensorElementType::Float32 => {
            DynAgent::Float32(Agent::<f32>::new(model, window_size, memory_dims))
        }
        TensorElementType::Int32 => {
            DynAgent::Int32(Agent::<i32>::new(model, window_size, memory_dims))
        }
        TensorElementType::Float64 => {
            DynAgent::Float64(Agent::<f64>::new(model, window_size, memory_dims))
        }
        TensorElementType::Int64 => {
            DynAgent::Int64(Agent::<i64>::new(model, window_size, memory_dims))
        }
        other => {
            return Err(AgentError::ModelDataType(ort::Error::new(format!(
                "Unsupported model input tensor type: {:?}",
                other
            ))));
        }
    };

    Ok(agent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_model() -> Model {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/model.onnx");
        crate::model::Loader::call(model_path)
            .expect("repository ONNX model should load")
            .1
    }

    #[test]
    fn new_by_type_constructs_the_requested_supported_variant() {
        let window_size = [5, 5];
        let memory_dims = [3, 3];

        assert!(matches!(
            new_by_type(
                TensorElementType::Float32,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(DynAgent::Float32(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Int32,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(DynAgent::Int32(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Float64,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(DynAgent::Float64(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Int64,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(DynAgent::Int64(_))
        ));
    }

    #[test]
    fn new_by_type_returns_an_error_for_unsupported_types() {
        let result = new_by_type(TensorElementType::Uint8, test_model(), [5, 5], [3, 3]);

        let Err(error) = result else {
            panic!("unsupported tensor types should be rejected");
        };
        assert_eq!(
            error.to_string(),
            "failed to create instance of Agent: Unsupported model input tensor type: Uint8"
        );
    }
}
