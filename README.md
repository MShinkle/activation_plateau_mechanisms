# Activation Plateau Mechanisms

This is a set of streamlined experiments described in the post *[todo](link-to-post)*. This includes scripts for interpolation-based elicitation of activation plateaus and additional analyses of model layers, components, splines, and jacobians.

*This work was performed as a part of the 2025 [PIBBSS summer research fellowship](https://pibbss.ai/fellowship/).*

## Requirements

Designed to be fairly dependency-light. You can install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Might not work with transformerlens 3.x.

## Usage

All scripts are in the `scripts/` directory:
- `python scripts/interpolate_and_record_activations.py `This generates activation data in the `activations/` directory that other scripts depend on. **Should be run first**.
- `spline_hamming_distances_plots.py` - Hamming distance analysis
- `step_sizes_resid_post_plots.py` - Step size analysis
- `relative_distances_layerwise_plots.py` - Layer-wise relative distances
- `relative_distances_logits_plots.py` - Logit-space relative distances
- `jacobians_attention.py` - Attention jacobian analysis
- `jacobians_mlp.py` - MLP jacobian analysis
- `jacobians_layerwise.py` - Layer-wise jacobian analysis
- `jacobians_full_residual.py` - Full residual stream jacobian analysis

You can also run the complete pipeline (activation extraction + all plots) via:
```bash
bash full_experiment.sh
```

## Configuration

You can edit `config.yaml` to configure which model, prompts, and number of steps to use. Currently just supports standard and layernorm-free gpt2 models, but should be very simple to accomodate any other transformerlens-supported model.