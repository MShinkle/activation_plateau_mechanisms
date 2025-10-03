#!/usr/bin/env python3
"""
Attention Jacobian Analysis: Compute and plot ∂(attn_out) / ∂(resid_pre) for each layer.
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


def compute_jacobian_attention(model, resid_pre_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for attention: ∂(attn_out) / ∂(resid_pre).
    Only computes jacobian for last token position wrt last token position.

    Args:
        resid_pre_interpolated: [n_steps, seq_len, hidden_dim]
        layer_idx: Which layer to compute jacobian for
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_pre_interpolated.shape[0]
    jacobians = []

    def forward_attention(last_token_resid):
        """Forward through ln1 + attention, last token only."""
        context = resid_pre_interpolated[step_idx, :-1].detach().to(device)
        full_resid = torch.cat([context, last_token_resid.unsqueeze(0)], dim=0)

        block = model.blocks[layer_idx]
        resid_with_batch = full_resid.unsqueeze(0)
        ln_out = block.ln1(resid_with_batch)
        attn_out = block.attn(ln_out, ln_out, ln_out)
        return attn_out[0, -1, :]

    for step_idx in range(n_steps):
        last_token = resid_pre_interpolated[step_idx, -1, :].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_attention, chunk_size=128)(last_token)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_attention(last_token)
            recorded_out = activations[f'layer{layer_idx}_attn_out'][step_idx, -1, :].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Attention validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

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

    print("\n=== Attention Jacobians ===")

    jacobian_norms_by_pair = {}
    jacobians_by_pair = {}

    for token_pair in TOKEN_PAIRS:
        pair_key = f"{token_pair[0]}_{token_pair[1]}"
        jacobian_norms_by_pair[pair_key] = {}
        jacobians_by_pair[pair_key] = []

        activations = load_activations(MODEL_NAME, SHARED_CONTEXT, 0, token_pair, N_STEPS)

        for layer_idx in tqdm(range(1, n_layers), desc=f"Computing Attention Jacobians for {token_pair}"):
            resid_pre = activations[f'layer{layer_idx-1}_resid_post']
            jacobians = compute_jacobian_attention(model, resid_pre, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            jacobian_norms_by_pair[pair_key][layer_idx] = norms
            jacobians_by_pair[pair_key].append(jacobians)

            del jacobians, resid_pre
            torch.cuda.empty_cache()

        del activations

    # Plot layerwise norms
    data_dict = {}
    for pair_key, layer_norms in jacobian_norms_by_pair.items():
        data_dict[pair_key] = {f"Layer {idx}": norms for idx, norms in layer_norms.items()}

    generate_interpolation_results_plot(
        data_dict=data_dict,
        suptitle="Jacobian Norms: Attention (∂attn_out / ∂resid_pre)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_attention_norms.png",
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
        suptitle="Jacobian Products: Attention (Full Chain)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_attention_product.png",
        n_steps=N_STEPS,
        shared_context=SHARED_CONTEXT,
        token_pairs=TOKEN_PAIRS
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
