//! Tool categories

use serde::{Deserialize, Serialize};

/// Tool category enum matching RIA spec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Code execution tools
    Execution,
    /// Testing tools
    Testing,
    /// Code analysis tools
    Analysis,
    /// Version control tools
    VersionControl,
    /// Build system tools
    Build,
    /// Search and navigation tools
    Search,
    /// Package management tools
    PackageManagement,
}

impl ToolCategory {
    /// Get the token ID range for this category (100,000-100,399)
    pub fn token_range(&self) -> (usize, usize) {
        match self {
            ToolCategory::Execution => (100_000, 100_056),
            ToolCategory::Testing => (100_057, 100_113),
            ToolCategory::Analysis => (100_114, 100_170),
            ToolCategory::VersionControl => (100_171, 100_227),
            ToolCategory::Build => (100_228, 100_284),
            ToolCategory::Search => (100_285, 100_341),
            ToolCategory::PackageManagement => (100_342, 100_399),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ToolCategory::Execution => "execution",
            ToolCategory::Testing => "testing",
            ToolCategory::Analysis => "analysis",
            ToolCategory::VersionControl => "version_control",
            ToolCategory::Build => "build",
            ToolCategory::Search => "search",
            ToolCategory::PackageManagement => "package_management",
        }
    }
}
