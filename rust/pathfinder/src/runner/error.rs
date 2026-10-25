use std::fmt;

use crate::agent::AgentError;
use crate::grid_env::CsvLoaderError;
use crate::model_loader::ModelLoaderError;

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
