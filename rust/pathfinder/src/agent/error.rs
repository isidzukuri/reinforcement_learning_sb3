use std::fmt;

#[derive(Debug)]
pub enum AgentError {
    ObservationTensor(ort::Error),
    Inference(ort::Error),
    OutputTensor(ort::Error),
    MissingModelOutput,
    NoValidActionValues,
    InvalidActionIndex(usize),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObservationTensor(error) => {
                write!(f, "failed to create observation tensor: {error}")
            }
            Self::Inference(error) => write!(f, "model inference failed: {error}"),
            Self::OutputTensor(error) => write!(f, "failed to read model output: {error}"),
            Self::MissingModelOutput => f.write_str("model returned no outputs"),
            Self::NoValidActionValues => {
                f.write_str("model output tensor contains no valid action values")
            }
            Self::InvalidActionIndex(index) => {
                write!(f, "model output action index {index} is out of bounds")
            }
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ObservationTensor(error) | Self::Inference(error) | Self::OutputTensor(error) => {
                Some(error)
            }
            Self::MissingModelOutput | Self::NoValidActionValues | Self::InvalidActionIndex(_) => {
                None
            }
        }
    }
}
