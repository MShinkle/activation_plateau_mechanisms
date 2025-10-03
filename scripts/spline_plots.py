#!/usr/bin/env python3
"""
Spline Analysis Script

This script computes Hamming distances of spline codes normalized by step size.
For each layer and step, compute Hamming distance between consecutive spline codes (mlp_post > 0),
normalized by the L2 step size in resid_mid space.
"""

import os
import sys
import torch
from tqdm import tqdm

sys.path.append('./scripts')
from utils import load_activations, load_config, generate_interpolation_results_plot

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']


def compute_normalized_hamming_distances(mlp_post_activations, resid_mid_activations):
    """Compute Hamming distances of spline codes normalized by resid_mid step sizes."""
    last_token_mlp = mlp_post_activations[:, -1, :]  # [n_steps, d_mlp]
    spline_codes = [(last_token_mlp[step] > 0).float() for step in range(last_token_mlp.shape[0])]

    normalized_distances = []
    for step in range(len(spline_codes) - 1):
        hamming_dist = torch.sum(spline_codes[step] != spline_codes[step + 1]).item()
        step_size = torch.norm(resid_mid_activations[step + 1, -1, :] - resid_mid_activations[step, -1, :]).item()
        normalized_distances.append(hamming_dist / step_size if step_size > 0 else 0.0)

    return normalized_distances

def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    # Load activations for all token pairs
    all_activations = [load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS) for token_pair in TOKEN_PAIRS]

    # Get number of layers
    n_layers = len([k for k in all_activations[0].keys() if k.startswith('layer') and k.endswith('_mlp_post')])

    # Compute hamming distances for each pair
    plot_data = {}
    for pair_idx, activations in enumerate(all_activations):
        layer_data = {}
        for layer_idx in tqdm(range(n_layers), desc=f"Pair {pair_idx + 1}/{len(all_activations)}"):
            mlp_post_layer = activations[f'layer{layer_idx}_mlp_post']
            resid_mid_layer = activations[f'layer{layer_idx}_resid_mid']
            distances = compute_normalized_hamming_distances(mlp_post_layer, resid_mid_layer)
            if distances:
                layer_data[f"Layer {layer_idx}"] = torch.tensor(distances)

        pair_name = f"{TOKEN_PAIRS[pair_idx][0]}_{TOKEN_PAIRS[pair_idx][1]}"
        plot_data[pair_name] = layer_data

    # Generate plot
    output_path = f"./plots/{MODEL_NAME}/spline_hamming_distances.png"
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Normalized Hamming Distances of Spline Codes",
        ylabel="Normalized Hamming Distance",
        output_path=output_path,
        n_steps=N_STEPS - 1,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS,
        alpha_range=[0, 1],
        skip_interpolation_layer=False
    )
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    main()
