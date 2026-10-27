use std::fmt;

use crate::agent::AgentError;
use crate::grid_env::CsvLoaderError;
use crate::model::ModelLoaderError;

#[derive(Debug)]
pub enum RunnerError {
    Agent(AgentError),
    CsvLoader(CsvLoaderError),
    ModelLoader(ModelLoaderError),
}

impl fmt::Display for RunnerError {
    fn fmt(&self, fm: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agent(error) => write!(fm, "{error}"),
            Self::CsvLoader(error) => write!(fm, "{error}"),
            Self::ModelLoader(error) => write!(fm, "{error}"),
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Agent(error) => Some(error),
            Self::CsvLoader(error) => Some(error),
            Self::ModelLoader(error) => Some(error),
        }
    }
}

impl From<AgentError> for RunnerError {
    fn from(error: AgentError) -> Self {
        Self::Agent(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn runner_errors_report_display_and_source() {
        let agent_error = AgentError::MissingModelOutput;
        let error = RunnerError::Agent(agent_error);
        assert_eq!(error.to_string(), "model returned no outputs");
        assert!(error.source().is_some());

        let csv_error = crate::grid_env::CsvLoaderError::EmptyOrInvalidGrid;
        let error = RunnerError::CsvLoader(csv_error);
        assert_eq!(error.to_string(), "Imported file is empty or invalid.");
        assert!(error.source().is_some());

        let model_error = ModelLoaderError::LoadModel(ort::Error::new("runtime model load issue"));
        let error = RunnerError::ModelLoader(model_error);
        assert_eq!(
            error.to_string(),
            "failed to load ONNX model: runtime model load issue"
        );
        assert!(error.source().is_some());
    }
}
