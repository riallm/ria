//! RIA Tools - Tool Integration Protocol (TIP) implementation
//!
//! Implements the tool protocol specified in the RIA spec:
//! - Tool call/result markers
//! - 7 tool categories
//! - Sandboxed execution
//! - Human approval workflows

pub mod categories;
pub mod executor;
pub mod protocol;
pub mod safety;

pub use categories::ToolCategory;
pub use executor::ToolExecutor;
pub use protocol::{ToolCall, ToolProtocol, ToolResult};
pub use safety::SafetyPolicy;
