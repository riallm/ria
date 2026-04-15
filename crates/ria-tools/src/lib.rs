//! RIA Tools - Tool Integration Protocol (TIP) implementation
//!
//! Implements the tool protocol specified in the RIA spec:
//! - Tool call/result markers
//! - 7 tool categories
//! - Sandboxed execution
//! - Human approval workflows

pub mod categories;
pub mod protocol;
pub mod executor;
pub mod safety;

pub use categories::ToolCategory;
pub use protocol::{ToolCall, ToolResult, ToolProtocol};
pub use executor::ToolExecutor;
pub use safety::SafetyPolicy;
