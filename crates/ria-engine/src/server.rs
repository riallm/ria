//! API server mode implementation

use anyhow::Result;
use ria_api::ApiServer;

use crate::Cli;

pub async fn run(cli: &Cli, host: String, port: u16) -> Result<()> {
    let model_name = cli.model.as_deref().unwrap_or("ria-8b").to_string();

    println!(
        "Starting RIA API Server on {}:{} (Model: {})",
        host, port, model_name
    );

    let server = ApiServer::new(host, port, model_name);
    server.start().await?;

    Ok(())
}
