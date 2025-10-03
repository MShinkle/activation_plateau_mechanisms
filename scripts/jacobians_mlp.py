#!/usr/bin/env python3
"""
MLP Jacobian Analysis: Compute and plot ∂(mlp_out) / ∂(resid_mid) for each layer.
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


def compute_jacobian_mlp(model, resid_mid_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for MLP: ∂(mlp_out) / ∂(resid_mid).

    Args:
        resid_mid_interpolated: [n_steps, hidden_dim] (last token only)
        layer_idx: Which layer to compute jacobian for
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    def forward_mlp(resid_mid):
        """Forward through ln2 + MLP."""
        block = model.blocks[layer_idx]
        resid_with_batch = resid_mid.unsqueeze(0).unsqueeze(0)
        mlp_out = block.mlp(block.ln2(resid_with_batch))
        return mlp_out.squeeze(0).squeeze(0)

    n_steps = resid_mid_interpolated.shape[0]
    jacobians = []

    for step_idx in range(n_steps):
        resid_mid = resid_mid_interpolated[step_idx].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_mlp, chunk_size=128)(resid_mid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_mlp(resid_mid)
            recorded_out = activations[f'layer{layer_idx}_mlp_out'][step_idx, -1, :].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"MLP validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid_mid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def main():
    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")

    model = load_model(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = model.cfg.n_layers
    print(f"Loaded {n_layers}-layer model on {device}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    print("\n=== MLP Jacobians ===")

    jacobian_norms_by_pair = {}
    jacobians_by_pair = {}

    for token_pair in TOKEN_PAIRS:
        pair_key = f"{token_pair[0]}_{token_pair[1]}"
        jacobian_norms_by_pair[pair_key] = {}
        jacobians_by_pair[pair_key] = []

        activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS)

        for layer_idx in tqdm(range(1, n_layers), desc=f"Computing MLP Jacobians for {token_pair}"):
            resid_mid = activations[f'layer{layer_idx}_resid_mid'][:, -1, :]
            jacobians = compute_jacobian_mlp(model, resid_mid, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            jacobian_norms_by_pair[pair_key][layer_idx] = norms
            jacobians_by_pair[pair_key].append(jacobians)

            del jacobians, resid_mid
            torch.cuda.empty_cache()

        del activations

    # Plot layerwise norms
    data_dict = {}
    for pair_key, layer_norms in jacobian_norms_by_pair.items():
        data_dict[pair_key] = {f"Layer {idx}": norms for idx, norms in layer_norms.items()}

    generate_interpolation_results_plot(
        data_dict=data_dict,
        suptitle="Jacobian Norms: MLP (∂mlp_out / ∂resid_mid)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_mlp_norms.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS,
        skip_interpolation_layer=False
    )

    # Compute and plot jacobian products
    product_norms = {}
    for pair_key, jacs_list in jacobians_by_pair.items():
        # Multiply jacobians across layers for each step
        product = jacs_list[0]  # Start with first layer
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)  # [n_steps, hidden_dim, hidden_dim]

        norms = torch.norm(product.view(product.shape[0], -1), dim=1)
        product_norms[pair_key] = norms

    generate_interpolation_results_plot(
        data_dict=product_norms,
        suptitle="Jacobian Products: MLP (Full Chain)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_mlp_product.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
