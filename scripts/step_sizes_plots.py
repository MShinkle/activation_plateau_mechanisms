#!/usr/bin/env python3
"""
Step Sizes Analysis Script

This script generates step size plots (L2 norm of differences between consecutive steps) 
for all variants: normal, freeze_attn, and freeze_mlp.
"""

import torch
import os
import sys
from typing import Dict, List
sys.path.append('./scripts')
from utils import load_config, load_activations, generate_interpolation_results_plot

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']


def compute_step_sizes(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
    """Compute step sizes (L2 norm of differences between consecutive steps) for each layer."""
    step_sizes = {}

    for key, layer_activations in activations.items():
        if key.startswith('layer') and key.endswith(f'_{hook_name}'):
            last_token = layer_activations[:, -1, :]  # [n_steps, hidden_dim]
            step_diffs = last_token[1:] - last_token[:-1]  # [n_steps-1, hidden_dim]
            step_norms = torch.norm(step_diffs, p=2, dim=1)  # [n_steps-1]

            layer_idx = int(key.split('_')[0].replace('layer', ''))
            step_sizes[f"Layer {layer_idx}"] = step_norms

    return step_sizes


def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    # Load activations and compute step sizes for each pair
    plot_data = {}
    for token_pair in TOKEN_PAIRS:
        activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS)
        pair_name = f"{token_pair[0]}_{token_pair[1]}"
        plot_data[pair_name] = compute_step_sizes(activations, 'resid_post')

    # Generate plot
    output_path = f"./plots/{MODEL_NAME}/step_sizes_resid_post.png"
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Resid Post Step Sizes",
        ylabel="L2 Norm of Step Difference",
        output_path=output_path,
        n_steps=N_STEPS - 1,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS,
        alpha_range=[0, 1]
    )
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
