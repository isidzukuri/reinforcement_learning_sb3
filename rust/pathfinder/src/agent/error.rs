use std::fmt;

#[derive(Debug)]
pub enum AgentError {
    Inference(ort::Error),
    OutputTensor(ort::Error),
    MissingModelOutput,
    NoValidActionValues,
    InvalidActionIndex(usize),
    ObservationTensor(ort::Error),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference(error) => write!(f, "model inference failed: {error}"),
            Self::OutputTensor(error) => write!(f, "failed to read model output: {error}"),
            Self::MissingModelOutput => f.write_str("model returned no outputs"),
            Self::NoValidActionValues => {
                f.write_str("model output tensor contains no valid action values")
            }
            Self::InvalidActionIndex(index) => {
                write!(f, "model output action index {index} is out of bounds")
            }
            Self::ObservationTensor(error) => {
                write!(f, "failed to create observation tensor: {error}")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn tensor_errors_report_display_and_source() {
        let error = AgentError::ObservationTensor(ort::Error::new("obs tensor issue"));
        assert_eq!(
            error.to_string(),
            "failed to create observation tensor: obs tensor issue"
        );
        assert!(error.source().is_some());

        let error = AgentError::Inference(ort::Error::new("inference issue"));
        assert_eq!(error.to_string(), "model inference failed: inference issue");
        assert!(error.source().is_some());

        let error = AgentError::OutputTensor(ort::Error::new("output tensor issue"));
        assert_eq!(
            error.to_string(),
            "failed to read model output: output tensor issue"
        );
        assert!(error.source().is_some());
    }

    #[test]
    fn non_tensor_errors_have_expected_messages_and_no_source() {
        let error = AgentError::MissingModelOutput;
        assert_eq!(error.to_string(), "model returned no outputs");
        assert!(error.source().is_none());

        let error = AgentError::NoValidActionValues;
        assert_eq!(
            error.to_string(),
            "model output tensor contains no valid action values"
        );
        assert!(error.source().is_none());

        let error = AgentError::InvalidActionIndex(7);
        assert_eq!(
            error.to_string(),
            "model output action index 7 is out of bounds"
        );
        assert!(error.source().is_none());
    }
}
