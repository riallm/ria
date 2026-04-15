//! Chat mode implementation

use std::io::Write;
use anyhow::Result;

use crate::Cli;

pub async fn run(cli: &Cli, system: Option<String>) -> Result<()> {
    let _system_prompt = system.unwrap_or_else(|| {
        "You are RIA (Rust Intelligence for Agentic Coding), \
         an AI coding assistant specialized in autonomous software development."
            .to_string()
    });

    println!("RIA Chat Mode (Model: {})", cli.model.as_deref().unwrap_or("none"));
    println!("Type 'quit' or 'exit' to leave.\n");

    // Chat loop
    loop {
        print!("> ");
        let mut input = String::new();
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input)?;

        let input = input.trim();
        if input == "quit" || input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Would generate response here
        println!("[Response would be generated for: {}]", input);
    }

    Ok(())
}
