//! Special token registry

use std::collections::HashMap;

/// Registry for special tokens
pub struct SpecialTokenRegistry {
    tokens: HashMap<String, usize>,
    reverse: HashMap<usize, String>,
    next_id: usize,
}

impl SpecialTokenRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tokens: HashMap::new(),
            reverse: HashMap::new(),
            next_id: 0,
        };

        // Register default special tokens
        registry.register("<|pad|>");
        registry.register("<|eos|>");
        registry.register("<|bos|>");
        registry.register("<|unk|>");
        registry.register("<|think|>");
        registry.register("<|plan_start|>");
        registry.register("<|code_start|>");
        registry.register("<|test_start|>");
        registry.register("<|debug_start|>");
        registry.register("<|tool_call|>");
        registry.register("<|tool_result|>");
        registry.register("<|verify|>");
        registry.register("<|file:|>");
        registry.register("<|error:|>");
        registry.register("<|success|>");
        registry.register("<|failure|>");

        registry
    }

    pub fn register(&mut self, token: &str) -> usize {
        let id = self.next_id;
        self.tokens.insert(token.to_string(), id);
        self.reverse.insert(id, token.to_string());
        self.next_id += 1;
        id
    }

    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.tokens.get(token).copied()
    }

    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.reverse.get(&id).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}
