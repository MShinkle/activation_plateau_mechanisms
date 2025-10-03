#!/usr/bin/env python3
"""
Logits Relative Distances Analysis Script

Generates logits relative distance plots for all freezing variants:
normal, freeze_attn, and freeze_mlp.
"""

import torch
import os
import sys
sys.path.append('./scripts')
from utils import load_activations, load_config, compute_relative_distances, construct_filepath, generate_interpolation_results_plot

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']

FREEZING_VARIANTS = [
    ("", "Normal"),
    ("_freeze_attn", "Attention Frozen"),
    ("_freeze_mlp", "MLP Frozen")
]


def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    for freeze_suffix, variant_name in FREEZING_VARIANTS:
        # Check if data exists for this variant
        if not os.path.exists(construct_filepath(MODEL_NAME, SHARED_CONTEXT, 0, TOKEN_PAIRS[0], N_STEPS, freeze_suffix)):
            print(f"\nSkipping {variant_name} (data not found)")
            continue

        print(f"\nProcessing {variant_name}...")

        # Load logits and compute relative distances
        plot_data = {}
        for token_pair in TOKEN_PAIRS:
            logits = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS, freeze_suffix)['logits']
            pair_name = f"{token_pair[0]}_{token_pair[1]}"
            plot_data[pair_name] = torch.tensor(compute_relative_distances(logits))

        # Generate plot
        output_path = f"./plots/{MODEL_NAME}/relative_distances_logits{freeze_suffix}.png"
        generate_interpolation_results_plot(
            data_dict=plot_data,
            suptitle=f"Logits Relative Distances ({variant_name})",
            ylabel="Relative Distance to A (0) vs B (1)",
            output_path=output_path,
            n_steps=N_STEPS,
            shared_context=SHARED_CONTEXT,
            token_pairs=TOKEN_PAIRS,
            alpha_range=[0, 1],
            skip_interpolation_layer=False
        )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()