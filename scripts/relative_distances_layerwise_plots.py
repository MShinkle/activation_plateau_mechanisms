#!/usr/bin/env python3
"""
Layerwise Relative Distances Analysis Script

Generates three types of relative distance plots:
1. Interpolate in layer 0, record in all subsequent layers
2. Interpolate in each layer, record in the last layer only
3. Interpolate in layer i, record in layer i+N for various N values
"""

import torch
import os
from typing import Dict, List
import sys
sys.path.append('./scripts')
from utils import load_activations, load_config, get_n_layers, compute_relative_distances, generate_interpolation_results_plot, construct_filepath

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']


def variant1_interpolate_layer0():
    """Variant 1: Interpolate in layer 0, record in all subsequent layers."""
    print("\nVariant 1: Interpolate in layer 0, record in all layers")

    plot_data = {}
    for token_pair in TOKEN_PAIRS:
        activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS)
        pair_name = f"{token_pair[0]}_{token_pair[1]}"
        layer_dict = {}
        for key, layer_activations in activations.items():
            if key.startswith('layer') and key.endswith('_resid_post'):
                layer_idx = int(key.split('_')[0].replace('layer', ''))
                layer_dict[f"Layer {layer_idx}"] = torch.tensor(compute_relative_distances(layer_activations))
        plot_data[pair_name] = layer_dict

    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Relative Distances (Interpolate in Layer 0)",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"./plots/{MODEL_NAME}/relative_distances_layerwise_layer0_interpolation.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS,
        alpha_range=[0, 1],
        skip_interpolation_layer=True
    )


def variant2_record_last_layer(n_layers: int):
    """Variant 2: Interpolate in each layer, record in last layer only."""
    print("\nVariant 2: Interpolate in each layer, record in last layer")

    last_layer_key = f'layer{n_layers - 1}_resid_post'
    plot_data = {}

    for pair_idx, token_pair in enumerate(TOKEN_PAIRS):
        pair_name = f"{token_pair[0]}_{token_pair[1]}"
        layer_dict = {}
        for interpolation_layer in range(n_layers - 1):
            activations = load_activations(MODEL_NAME, SHARED_CONTEXT, interpolation_layer, token_pair, N_STEPS)
            distances = compute_relative_distances(activations[last_layer_key])
            layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
        plot_data[pair_name] = layer_dict

    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Relative Distances (Record in Last Layer)",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"./plots/{MODEL_NAME}/relative_distances_layerwise_last_layer_recording.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS,
        alpha_range=[0, 1],
        skip_interpolation_layer=True
    )


def variant3_record_layer_plus_n(n_layers: int):
    """Variant 3: Interpolate in layer i, record in layer i+N for various N."""
    print("\nVariant 3: Interpolate in layer i, record in layer i+N")

    for N in [1, 4, 8, 16, 24]:
        print(f"  Processing N={N}...")

        plot_data = {}
        for token_pair in TOKEN_PAIRS:
            pair_name = f"{token_pair[0]}_{token_pair[1]}"
            layer_dict = {}
            for interpolation_layer in range(n_layers - N):
                activations = load_activations(MODEL_NAME, SHARED_CONTEXT, interpolation_layer, token_pair, N_STEPS)
                target_layer_key = f'layer{interpolation_layer + N}_resid_post'
                distances = compute_relative_distances(activations[target_layer_key])
                layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
            plot_data[pair_name] = layer_dict

        generate_interpolation_results_plot(
            data_dict=plot_data,
            suptitle=f"Relative Distances (N={N})",
            ylabel="Relative Distance to Token A (0) vs Token B (1)",
            output_path=f"./plots/{MODEL_NAME}/relative_distances_layerwise_N{N}.png",
            n_steps=N_STEPS,
            shared_context=SHARED_CONTEXT,
            token_pairs=TOKEN_PAIRS,
            alpha_range=[0, 1],
            skip_interpolation_layer=True
        )


def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    # Check if data exists
    if not os.path.exists(construct_filepath(MODEL_NAME, SHARED_CONTEXT, 0, TOKEN_PAIRS[0], N_STEPS)):
        print("Data not found, skipping")
        return

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, TOKEN_PAIRS[0], N_STEPS)
    n_layers = get_n_layers(activations)
    print(f"Model has {n_layers} layers")

    variant1_interpolate_layer0()
    variant2_record_last_layer(n_layers)
    variant3_record_layer_plus_n(n_layers)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
