#!/usr/bin/env python3
"""
Full Residual Jacobian Analysis: Compute and plot ∂(layer_N resid_post) / ∂(layer_0 resid_post).
"""

import torch
import os
from tqdm import tqdm
import sys
sys.path.append('./scripts')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']


def compute_jacobian_full_residual(model, resid_post_interpolated: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute jacobian from layer 0 to last layer.
    Only computes jacobian for last token position wrt last token position.

    Args:
        resid_post_interpolated: [n_steps, seq_len, hidden_dim]

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_interpolated.shape[0]
    jacobians = []

    def forward_through_all_layers(last_token_resid):
        """Forward from layer 0 through all layers, last token only."""
        context = resid_post_interpolated[step_idx, :-1].detach().to(device)
        full_resid = torch.cat([context, last_token_resid.unsqueeze(0)], dim=0)

        activation = full_resid.unsqueeze(0)
        for layer_idx in range(model.cfg.n_layers):
            activation = model.blocks[layer_idx](activation)
        return activation[0, -1, :]

    for step_idx in range(n_steps):
        last_token = resid_post_interpolated[step_idx, -1, :].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_through_all_layers, chunk_size=128)(last_token)

        jacobians.append(jac.detach().cpu())
        del jac, last_token
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    model = load_model(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = model.cfg.n_layers
    print(f"Loaded {n_layers}-layer model on {device}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    print("\n=== Full Residual Stream Jacobians ===")

    jacobian_norms = {}

    for token_pair in tqdm(TOKEN_PAIRS, desc="Computing Full Residual Jacobians"):
        activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS)
        resid_post_layer0 = activations['layer0_resid_post']

        jacobians = compute_jacobian_full_residual(model, resid_post_layer0, device)

        norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)

        pair_key = f"{token_pair[0]}_{token_pair[1]}"
        jacobian_norms[pair_key] = norms

        del jacobians, resid_post_layer0, activations
        torch.cuda.empty_cache()

    # Plot
    generate_interpolation_results_plot(
        data_dict=jacobian_norms,
        suptitle="Jacobian Norms: Full Residual Stream (Layer 0 → Last Layer)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_full_residual_norms.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
