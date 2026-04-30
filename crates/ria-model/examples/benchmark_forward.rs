use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use ria_core::config::{ModelConfig, RecurrentDepthConfig};
use ria_model::RiaModel;

fn main() -> Result<()> {
    let iterations = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let steps = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);

    let recurrent = RecurrentDepthConfig {
        prelude_layers: 1,
        recurrent_iterations: iterations,
        min_recurrent_iterations: 1,
        max_recurrent_iterations: 64,
        coda_layers: 1,
        residual_scale: 1.0,
    };
    let config = ModelConfig::custom(64, 64, 4, 128, 4, 2, 512).with_recurrent_depth(recurrent);
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
    let model = RiaModel::new(vb, &config)?;
    let input = Tensor::from_vec((0..32).map(|id| id as u32).collect(), (1, 32), &Device::Cpu)?;

    for _ in 0..3 {
        let _ = model.forward_with_recurrent_iterations(&input, iterations)?;
    }

    let start = Instant::now();
    for _ in 0..steps {
        let _ = model.forward_with_recurrent_iterations(&input, iterations)?;
    }
    let elapsed = start.elapsed();
    let per_forward = elapsed.as_secs_f64() / steps as f64;

    println!(
        "ria-model recurrent forward: iterations={}, steps={}, {:.6}s/forward",
        iterations, steps, per_forward
    );

    Ok(())
}
