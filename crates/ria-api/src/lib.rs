//! RIA API - OpenAI-compatible REST API
//!
//! Implements the API endpoints specified in the RIA spec:
//! - OpenAI-compatible endpoints (/v1/completions, /v1/chat/completions)
//! - RIA-specific endpoints (/v1/code/complete, /v1/code/debug, etc.)
//! - Agentic workflows (/v1/agent/execute)

pub mod handlers;
pub mod models;
pub mod routes;
pub mod server;

pub use server::ApiServer;
