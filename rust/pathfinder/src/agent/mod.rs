mod agent;
mod error;
mod tensor_layer;

pub use agent::{Agent, AgentInner};
pub use error::AgentError;
pub use tensor_layer::TensorLayer;

use ort::session::Session as Model;
use ort::tensor::TensorElementType;

pub fn new_by_type(
    data_type: TensorElementType,
    model: Model,
    window_size: [usize; 2],
    memory_dims: [usize; 2],
) -> Result<Agent, AgentError> {
    let agent = match data_type {
        TensorElementType::Float32 => {
            Agent::Float32(AgentInner::<f32>::new(model, window_size, memory_dims))
        }
        TensorElementType::Int32 => {
            Agent::Int32(AgentInner::<i32>::new(model, window_size, memory_dims))
        }
        TensorElementType::Float64 => {
            Agent::Float64(AgentInner::<f64>::new(model, window_size, memory_dims))
        }
        TensorElementType::Int64 => {
            Agent::Int64(AgentInner::<i64>::new(model, window_size, memory_dims))
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
            Ok(Agent::Float32(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Int32,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(Agent::Int32(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Float64,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(Agent::Float64(_))
        ));
        assert!(matches!(
            new_by_type(
                TensorElementType::Int64,
                test_model(),
                window_size,
                memory_dims
            ),
            Ok(Agent::Int64(_))
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
