#!/bin/bash

# Full experiment script for activation plateau analyses

echo "Running activation extraction (normal)..."
python ./scripts/interpolate_and_record_activations.py

echo "Running activation extraction (attention frozen)..."
python ./scripts/interpolate_and_record_activations.py --freeze_attention

echo "Running activation extraction (MLP frozen)..."
python ./scripts/interpolate_and_record_activations.py --freeze_mlp

echo "Generating plots..."
python ./scripts/spline_plots.py
python ./scripts/step_sizes_plots.py
python ./scripts/relative_distances_layerwise_plots.py
python ./scripts/relative_distances_logits_plots.py
python ./scripts/jacobians_attention.py
python ./scripts/jacobians_mlp.py
python ./scripts/jacobians_layerwise.py
python ./scripts/jacobians_full_residual.py

echo "All plots completed! Check: ./plots/"
