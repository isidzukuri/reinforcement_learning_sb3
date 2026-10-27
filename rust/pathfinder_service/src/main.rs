use std::{
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    body::to_bytes,
    extract::{FromRequest, Multipart, Request, State},
    http::{StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::post,
};
use pathfinder::{
    agent::{Agent, SharedModel},
    grid_env::CsvLoader,
    model_loader,
    runner::{AGENT_WINDOW_SIZE, Runner},
};
use serde::Serialize;
use tokio::sync::Semaphore;

const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const SESSION_INTRA_THREADS: usize = 1;

#[derive(Clone)]
struct AppState {
    sessions: Arc<SessionPool>,
    request_slots: Arc<Semaphore>,
}

struct SessionPool {
    available: Mutex<Vec<SharedModel>>,
}

struct SessionLease {
    pool: Arc<SessionPool>,
    model: Option<SharedModel>,
}

impl SessionPool {
    fn load(model_path: &PathBuf, pool_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let model_bytes = std::fs::read(model_path)?;
        let models = (0..pool_size)
            .map(|_| {
                model_loader::from_bytes_with_intra_threads(&model_bytes, SESSION_INTRA_THREADS)
                    .map(|session| Arc::new(Mutex::new(session)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            available: Mutex::new(models),
        })
    }

    fn acquire(self: &Arc<Self>) -> Result<SessionLease, String> {
        let model = self
            .available
            .lock()
            .map_err(|error| format!("session pool lock failed: {error}"))?
            .pop()
            .ok_or_else(|| {
                "no ONNX session is available despite an acquired request slot".to_owned()
            })?;

        Ok(SessionLease {
            pool: Arc::clone(self),
            model: Some(model),
        })
    }

    #[cfg(test)]
    fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }
}

impl SessionLease {
    fn model(&self) -> SharedModel {
        Arc::clone(
            self.model
                .as_ref()
                .expect("session lease always contains a model"),
        )
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        if let Some(model) = self.model.take() {
            let mut available = self
                .pool
                .available
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            available.push(model);
        }
    }
}

#[derive(Serialize)]
struct SolveResponse {
    success: bool,
    errors: Vec<String>,
    path: Vec<[usize; 2]>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::var_os("PATHFINDER_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model.onnx"));
    let pool_size = session_pool_size()?;
    let sessions = Arc::new(SessionPool::load(&model_path, pool_size)?);
    println!("Loaded {pool_size} ONNX sessions");
    let state = AppState {
        sessions,
        request_slots: Arc::new(Semaphore::new(pool_size)),
    };
    let app = Router::new().route("/solve", post(solve)).with_state(state);
    let address: SocketAddr = env::var("PATHFINDER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3005".to_owned())
        .parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("Pathfinder service listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

fn session_pool_size() -> Result<usize, Box<dyn std::error::Error>> {
    let pool_size = match env::var("PATHFINDER_SESSION_POOL_SIZE") {
        Ok(value) => value.parse::<usize>()?,
        Err(env::VarError::NotPresent) => std::thread::available_parallelism()?.get(),
        Err(error) => return Err(error.into()),
    };

    if pool_size == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "PATHFINDER_SESSION_POOL_SIZE must be greater than zero",
        )
        .into());
    }

    Ok(pool_size)
}

async fn solve(State(state): State<AppState>, request: Request) -> Response {
    let csv = match read_csv(request).await {
        Ok(csv) => csv,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(SolveResponse {
                    success: false,
                    errors: vec![error],
                    path: Vec::new(),
                }),
            )
                .into_response();
        }
    };

    let request_slot = match Arc::clone(&state.request_slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(tokio::sync::TryAcquireError::NoPermits) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(SolveResponse {
                    success: false,
                    errors: vec!["all inference sessions are busy; retry later".to_owned()],
                    path: Vec::new(),
                }),
            )
                .into_response();
        }
        Err(tokio::sync::TryAcquireError::Closed) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SolveResponse {
                    success: false,
                    errors: vec!["inference request slots are unavailable".to_owned()],
                    path: Vec::new(),
                }),
            )
                .into_response();
        }
    };

    match tokio::task::spawn_blocking(move || {
        let _request_slot = request_slot;
        solve_grid(&state, &csv)
    })
    .await
    {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(error)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(SolveResponse {
                success: false,
                errors: vec![error],
                path: Vec::new(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SolveResponse {
                success: false,
                errors: vec![format!("solver task failed: {error}")],
                path: Vec::new(),
            }),
        )
            .into_response(),
    }
}

async fn read_csv(request: Request) -> Result<String, String> {
    let is_multipart = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| {
            content_type
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        });

    if is_multipart {
        let mut multipart = Multipart::from_request(request, &())
            .await
            .map_err(|error| format!("invalid multipart request: {error}"))?;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| format!("invalid multipart field: {error}"))?
        {
            if field
                .name()
                .is_some_and(|name| name != "file" && name != "csv")
            {
                continue;
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|error| format!("failed to read CSV upload: {error}"))?;
            if bytes.len() > MAX_REQUEST_BYTES {
                return Err("CSV upload exceeds the 2 MiB limit".to_owned());
            }
            return String::from_utf8(bytes.to_vec())
                .map_err(|_| "CSV request body must be valid UTF-8".to_owned());
        }
        return Err("multipart request must include a 'file' or 'csv' field".to_owned());
    }

    let bytes = to_bytes(request.into_body(), MAX_REQUEST_BYTES)
        .await
        .map_err(|error| format!("failed to read CSV request body: {error}"))?;
    String::from_utf8(bytes.to_vec()).map_err(|_| "CSV request body must be valid UTF-8".to_owned())
}

fn solve_grid(state: &AppState, csv: &str) -> Result<SolveResponse, String> {
    let grid = CsvLoader::load_from(std::io::Cursor::new(csv.as_bytes()))
        .map_err(|error| format!("invalid CSV grid: {error}"))?;
    let session = state.sessions.acquire()?;
    let agent = Agent::new_with_shared_model(
        session.model(),
        AGENT_WINDOW_SIZE,
        [grid.width, grid.height],
    );
    let mut runner = Runner::new(grid, agent);

    let solution = runner.call();
    drop(runner);
    drop(session);

    match solution {
        Ok(solution) => Ok(SolveResponse {
            success: solution.success,
            errors: Vec::new(),
            path: solution.path,
        }),
        Err(error) => Ok(SolveResponse {
            success: false,
            errors: vec![error.to_string()],
            path: Vec::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn csv_reader_parses_raw_csv() {
        let grid = CsvLoader::load_from(std::io::Cursor::new("# start: 0, 0\n0,1\n")).unwrap();

        assert_eq!([grid.width, grid.height], [2, 1]);
        assert_eq!(grid.start, [0, 0]);
    }

    #[test]
    fn response_uses_the_expected_json_fields() {
        let response = SolveResponse {
            success: true,
            errors: Vec::new(),
            path: vec![[2, 3]],
        };

        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["errors"], serde_json::json!([]));
        assert_eq!(json["path"], serde_json::json!([[2, 3]]));
    }

    #[test]
    fn sessions_are_reused_after_a_lease_is_dropped() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model.onnx");
        let pool = Arc::new(SessionPool::load(&model_path, 1).unwrap());

        let lease = pool.acquire().unwrap();
        assert_eq!(pool.available_count(), 0);
        drop(lease);

        assert_eq!(pool.available_count(), 1);
    }
}
