# ria

Reactive Intelligence Architecture.

## Model Architecture

The `ria-model` crate implements the RIA decoder block from `ria-spec`:

- RMSNorm pre-normalization
- Grouped Query Attention with RoPE
- Tool Integration Router
- Dual-path SwiGLU feed-forward layers
- Residual connections
- Tied token embedding and language-modeling head

It also supports the recurrent-depth variant requested for RIA experiments:

```text
Token Embedding
Prelude blocks
Shared RecurrentBlock repeated N times
Coda blocks
Final RMSNorm
Tied LM head
```

The recurrent loop count is configurable at inference time, while the recurrent
core shares one parameterized block across all iterations.

## Rust Usage

```rust
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use ria_core::config::{ModelConfig, RecurrentDepthConfig};
use ria_model::{causal_lm_loss, RiaModel};

# fn main() -> candle_core::Result<()> {
let recurrent = RecurrentDepthConfig {
    prelude_layers: 1,
    recurrent_iterations: 4,
    min_recurrent_iterations: 1,
    max_recurrent_iterations: 16,
    coda_layers: 1,
    residual_scale: 1.0,
};
let config = ModelConfig::custom(16, 8, 4, 16, 2, 1, 32)
    .with_recurrent_depth(recurrent);
let var_map = VarMap::new();
let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
let model = RiaModel::new(vb, &config)?;

let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu)?;
let logits = model.forward_with_recurrent_iterations(&input, 6)?;
let loss = causal_lm_loss(&logits, &input)?;
# Ok(())
# }
```

## Checks

```bash
cargo test -p ria-model
cargo run -p ria-model --example benchmark_forward -- 4 20
```
