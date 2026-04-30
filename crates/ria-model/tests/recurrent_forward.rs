use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use ria_core::config::{ModelConfig, RecurrentDepthConfig};
use ria_model::{causal_lm_loss, RiaModel};

fn var_builder() -> VarBuilder<'static> {
    let var_map = Box::leak(Box::new(VarMap::new()));
    VarBuilder::from_varmap(var_map, DType::F32, &Device::Cpu)
}

#[test]
fn recurrent_model_forward_and_lm_loss() -> Result<()> {
    let recurrent = RecurrentDepthConfig {
        prelude_layers: 1,
        recurrent_iterations: 2,
        min_recurrent_iterations: 1,
        max_recurrent_iterations: 4,
        coda_layers: 1,
        residual_scale: 0.75,
    };
    let config = ModelConfig::custom(16, 8, 4, 16, 2, 1, 32).with_recurrent_depth(recurrent);
    let model = RiaModel::new(var_builder(), &config)?;
    let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu)?;

    let logits = model.forward_with_recurrent_iterations(&input, 3)?;
    assert_eq!(logits.dims(), &[1, 4, 32]);

    let loss = causal_lm_loss(&logits, &input)?;
    assert!(loss.to_scalar::<f32>()?.is_finite());

    Ok(())
}
