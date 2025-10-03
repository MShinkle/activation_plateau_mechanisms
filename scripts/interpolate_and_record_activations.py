#!/usr/bin/env python3
"""
Interpolate activations between token pairs in transformer models.

For each token pair and each layer:
1. Collect resid_post activations for both tokens at all layers
2. Interpolate between them at the specified layer using SLERP
3. Inject interpolated activations and propagate through subsequent layers
4. Record activations (attn_out, resid_mid, mlp_post, mlp_out, resid_post) and logits

Outputs: One file per (token_pair, interpolation_layer, freeze_mode) combination
"""

import torch
import os
import argparse
from typing import Dict
from tqdm import tqdm
from transformer_lens import HookedTransformer
import sys
sys.path.append('./scripts')
from utils import load_model, load_config, slerp_rescale, construct_filepath

config = load_config()
MODEL_NAME = config['model_name']
SHARED_CONTEXT = config['shared_context']
TOKEN_PAIRS = config['token_pairs']
N_STEPS = config['n_steps']

# Hook types to record at each layer
ACTIVATION_HOOKS = [
    ('attn_out', 'blocks.{}.hook_attn_out'),
    ('resid_mid', 'blocks.{}.hook_resid_mid'),
    ('mlp_post', 'blocks.{}.mlp.hook_post'),
    ('mlp_out', 'blocks.{}.hook_mlp_out'),
    ('resid_post', 'blocks.{}.hook_resid_post'),
]


def collect_original_activations(model: HookedTransformer, shared_context: str, token: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect resid_post activations at all layers for a given token.

    Returns dict with keys like 'layer{i}_resid_post' -> tensor of shape [1, hidden_dim]
    """
    full_sequence = f"{shared_context} {token}"
    tokens = model.to_tokens(full_sequence, prepend_bos=False)

    activations = {}

    def create_collection_hook(layer_idx):
        def hook_fn(activation, hook):
            activations[f'layer{layer_idx}_resid_post'] = activation[:, -1, :].cpu().clone()
            return activation
        return hook_fn

    hooks = [(f'blocks.{i}.hook_resid_post', create_collection_hook(i))
             for i in range(model.cfg.n_layers)]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    return activations


def interpolate_resid_post_layer(
    model: HookedTransformer,
    shared_context: str,
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
    freeze_attention: bool = False,
    freeze_mlp: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Interpolate resid_post at specified layer and record all downstream activations.

    Returns dict with keys like 'layer{i}_{hook_name}' -> tensors of shape [n_steps, seq_len, hidden_dim]
    and 'logits' -> tensor of shape [n_steps, seq_len, vocab_size]
    """
    # Get resid_post activations at the interpolation layer for both tokens
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]

    # Compute SLERP interpolations on device
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)

    interpolated_resid_post = torch.stack([
        slerp_rescale(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim] on device

    # Free up device memory immediately
    del resid_a_device, resid_b_device, alphas

    # Storage for collected activations (keep interpolated values on device for injection)
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_resid_post.cpu().clone()}

    # Hook to inject interpolated activations (already on device, no transfer needed)
    def inject_hook(activation, hook):
        # interpolated_resid_post is [n_steps, hidden_dim], activation is [n_steps, seq_len, hidden_dim]
        activation[:, -1, :] = interpolated_resid_post
        return activation

    # Hook to collect and optionally freeze activations
    def create_collection_hook(hook_name, target_layer):
        def hook_fn(activation, hook):
            # Freeze to mean across all steps if requested
            if freeze_attention and hook_name == 'attn_out':
                mean_activation = activation[:, -1, :].mean(dim=0, keepdim=True)
                activation[:, -1, :] = mean_activation.expand(activation.shape[0], -1)
            elif freeze_mlp and hook_name == 'mlp_out':
                mean_activation = activation[:, -1, :].mean(dim=0, keepdim=True)
                activation[:, -1, :] = mean_activation.expand(activation.shape[0], -1)

            activations[f'layer{target_layer}_{hook_name}'] = activation.cpu().clone()
            return activation
        return hook_fn

    # Build hooks list
    hooks = [(f'blocks.{interpolation_layer}.hook_resid_post', inject_hook)]

    for target_layer in range(interpolation_layer, model.cfg.n_layers):
        for hook_name, hook_pattern in ACTIVATION_HOOKS:
            hooks.append((hook_pattern.format(target_layer), create_collection_hook(hook_name, target_layer)))

    # Create input tokens (context + dummy token that gets replaced)
    context_tokens = model.to_tokens(shared_context, prepend_bos=False)
    full_tokens = torch.cat([context_tokens[0], torch.tensor([0], device=device)])
    batched_tokens = full_tokens.unsqueeze(0).repeat(n_steps, 1)

    # Run forward pass with all interpolated activations in parallel
    with torch.no_grad():
        logits = model.run_with_hooks(batched_tokens, fwd_hooks=hooks, return_type="logits")

    activations['logits'] = logits.cpu().clone()

    # Clean up device memory
    del interpolated_resid_post, logits

    return activations


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--freeze_attention', action='store_true', help='Freeze attention outputs to first step')
    parser.add_argument('--freeze_mlp', action='store_true', help='Freeze MLP outputs to first step')
    args = parser.parse_args()

    print(f"Model: {MODEL_NAME} | Context: '{SHARED_CONTEXT}' | Steps: {N_STEPS}")
    print(f"Freeze attention: {args.freeze_attention} | Freeze MLP: {args.freeze_mlp}")

    model = load_model(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = model.cfg.n_layers
    print(f"Loaded {n_layers}-layer model on {device}")

    output_dir = f"./activations/{MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    for token_pair in TOKEN_PAIRS:
        print(f"\nProcessing {token_pair}")

        # Collect reference activations for both tokens
        reference_activations = {}
        for idx, token in enumerate(token_pair):
            reference_activations[f'token_{idx}'] = collect_original_activations(
                model, SHARED_CONTEXT, token, device
            )

        # Interpolate at each layer
        freeze_suffix = ""
        if args.freeze_attention:
            freeze_suffix = "_freeze_attn"
        elif args.freeze_mlp:
            freeze_suffix = "_freeze_mlp"

        pbar = tqdm(range(n_layers), desc=f"Interpolating {token_pair}{freeze_suffix}")
        for interpolation_layer in pbar:
            interpolated_activations = interpolate_resid_post_layer(
                model=model,
                shared_context=SHARED_CONTEXT,
                resid_post_a=reference_activations['token_0'],
                resid_post_b=reference_activations['token_1'],
                interpolation_layer=interpolation_layer,
                n_steps=N_STEPS,
                device=device,
                freeze_attention=args.freeze_attention,
                freeze_mlp=args.freeze_mlp
            )

            # Save to disk
            filepath = construct_filepath(MODEL_NAME, SHARED_CONTEXT, interpolation_layer, token_pair, N_STEPS, freeze_suffix)
            torch.save(interpolated_activations, filepath)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()