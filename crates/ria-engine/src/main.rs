//! rial lm - RIA LLM Inference Engine
//!
//! CLI for running the RIA LLM model with support for:
//! - Interactive chat mode
//! - Code completion
//! - Debug assistance
//! - Agentic workflows
//! - API server mode

use clap::{Parser, Subcommand};
use ria_core::{device::Backend, generation::GenerationConfig};

mod chat;
mod complete;
mod debug;
mod server;

/// RIA LLM Inference Engine
#[derive(Parser, Debug)]
#[command(name = "rial lm", about = "RIA LLM Inference Engine", version)]
struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<String>,

    /// Backend device type
    #[arg(short, long, default_value = "cpu")]
    device: Backend,

    /// Maximum sequence length
    #[arg(long, default_value = "4096")]
    max_seq_len: usize,

    /// Enable memory-mapped loading
    #[arg(long, default_value = "true")]
    mmap: bool,

    /// Number of layers to prefetch
    #[arg(long, default_value = "4")]
    prefetch_layers: usize,

    /// Cache size in bytes
    #[arg(long, default_value = "268435456")] // 256MB
    cache_size_bytes: u64,

    /// Enable profiling
    #[arg(long, default_value = "false")]
    profile: bool,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Interactive chat mode
    Chat {
        /// System prompt
        #[arg(long)]
        system: Option<String>,
    },
    /// Code completion
    Complete {
        /// File path
        #[arg(short, long)]
        file: String,
        /// Cursor position
        #[arg(long, default_value = "0")]
        position: usize,
    },
    /// Debug assistance
    Debug {
        /// Code to debug
        #[arg(short, long)]
        code: String,
        /// Error message
        #[arg(short, long)]
        error: Option<String>,
    },
    /// Start API server
    Serve {
        /// Host address
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port
        #[arg(long, default_value = "8000")]
        port: u16,
    },
    /// Generate text
    Generate {
        /// Prompt text
        #[arg(short, long)]
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Temperature
        #[arg(long, default_value = "0.7")]
        temperature: f64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();

    // Set up logging level based on verbosity
    if cli.verbose > 0 {
        tracing::info!("Verbose mode enabled");
    }

    match &cli.command {
        Some(Commands::Chat { system }) => {
            chat::run(&cli, system.clone()).await?;
        }
        Some(Commands::Complete { file, position }) => {
            complete::run(&cli, file.clone(), *position).await?;
        }
        Some(Commands::Debug { code, error }) => {
            debug::run(&cli, code.clone(), error.clone()).await?;
        }
        Some(Commands::Serve { host, port }) => {
            server::run(&cli, host.clone(), *port).await?;
        }
        Some(Commands::Generate {
            prompt,
            max_tokens,
            temperature,
        }) => {
            let config = GenerationConfig {
                max_new_tokens: *max_tokens,
                temperature: *temperature,
                ..Default::default()
            };
            tracing::info!("Generating with config: {:?}", config);
            tracing::info!("Prompt: {}", prompt);
        }
        None => {
            // Default to chat mode
            chat::run(&cli, None).await?;
        }
    }

    Ok(())
}
