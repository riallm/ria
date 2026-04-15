//! Route configuration

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;

use crate::handlers::*;

/// Create the API router with all routes
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::permissive();

    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // RIA code-specific endpoints
        .route("/v1/code/complete", post(code_complete))
        .route("/v1/code/debug", post(code_debug))
        // RIA agentic endpoints
        .route("/v1/agent/execute", post(agent_execute))
        .with_state(state)
        .layer(cors)
}
