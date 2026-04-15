//! API server

use axum::Router;
use ria_core::RiaResult;
use tokio::net::TcpListener;

use crate::routes;
use crate::handlers::AppState;

/// API server configuration
pub struct ApiServer {
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

impl ApiServer {
    pub fn new(host: String, port: u16, model_name: String) -> Self {
        Self {
            host,
            port,
            model_name,
        }
    }

    /// Start the API server
    pub async fn start(&self) -> RiaResult<()> {
        let state = AppState {
            model_name: self.model_name.clone(),
        };

        let app = routes::create_router(state);

        let addr = format!("{}:{}", self.host, self.port);
        let listener = TcpListener::bind(&addr).await?;

        tracing::info!("RIA API server listening on {}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}
