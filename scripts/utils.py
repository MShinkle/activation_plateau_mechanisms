#!/usr/bin/env python3
"""
Utility functions for modular analyses.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Union

# Dictionary of no-LayerNorm models with standardized names
NO_LAYERNORM_MODELS = {
    "gpt2-small_LNFree": "schaeff/gpt2-small_LNFree300",
    "gpt2-medium_LNFree": "schaeff/gpt2-medium_LNFree500", 
    "gpt2-large_LNFree": "schaeff/gpt2-large_LNFree600",
    "gpt2-xl_LNFree": "schaeff/gpt2-xl_LNFree800"
}

def load_gpt2_regular(model_name):
    """Load regular GPT-2 with LayerNorm."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Map model names to actual model names
    model_mapping = {
        "gpt2-small": "gpt2",
        "gpt2-medium": "gpt2-medium", 
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl"
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown regular model: {model_name}. Available: {list(model_mapping.keys())}")
    
    actual_model_name = model_mapping[model_name]
    model = HookedTransformer.from_pretrained(actual_model_name, fold_ln=False, center_unembed=False).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(actual_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device, model_name

def load_gpt2_no_ln(model_name):
    """Load GPT-2 without LayerNorm from the specified model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name not in NO_LAYERNORM_MODELS:
        raise ValueError(f"Unknown no-LayerNorm model: {model_name}. Available: {list(NO_LAYERNORM_MODELS.keys())}")
    
    hf_model_path = NO_LAYERNORM_MODELS[model_name]
    
    # Load the HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_path).to("cpu")
    
    # Undo hacky LayerNorm removal
    for block in hf_model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    hf_model.transformer.ln_f.weight.data = hf_model.transformer.ln_f.weight.data / 1e6
    hf_model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    def removeLN(transformer_lens_model):
        for i in range(len(transformer_lens_model.blocks)):
            transformer_lens_model.blocks[i].ln1 = torch.nn.Identity()
            transformer_lens_model.blocks[i].ln2 = torch.nn.Identity()
        transformer_lens_model.ln_final = torch.nn.Identity()
    
    # Determine the base model size for TransformerLens
    if "small" in model_name:
        base_model = "gpt2"
    elif "medium" in model_name:
        base_model = "gpt2-medium"
    elif "large" in model_name:
        base_model = "gpt2-large"
    elif "xl" in model_name:
        base_model = "gpt2-xl"
    
    model = HookedTransformer.from_pretrained(base_model, hf_model=hf_model, fold_ln=True, center_unembed=False).to(device)
    removeLN(model)
    model.cfg.normalization_type = None
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device, model_name

def load_model(model_name):
    """Load either regular, no-LayerNorm GPT-2, or OPT model."""
    if 'LNFree' in model_name:
        model, tokenizer, device, actual_name = load_gpt2_no_ln(model_name)
    else:
        model, tokenizer, device, actual_name = load_gpt2_regular(model_name)
    # Store the original model name for saving
    model.original_model_name = model_name
    return model

def slerp_rescale(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation with norm rescaling.
    Interpolates angle evenly, interpolates magnitude linearly.
    """
    # Get norms for rescaling
    norm_v0 = torch.norm(v0, dim=-1, keepdim=True)
    norm_v1 = torch.norm(v1, dim=-1, keepdim=True)

    # Normalize vectors
    v0_norm = v0 / norm_v0
    v1_norm = v1 / norm_v1

    # Compute angle between normalized vectors
    dot_product = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(torch.abs(dot_product))

    # SLERP computation
    sin_theta = torch.sin(theta)
    weight_v0 = torch.sin((1 - t) * theta) / sin_theta
    weight_v1 = torch.sin(t * theta) / sin_theta
    slerp_result = weight_v0 * v0_norm + weight_v1 * v1_norm

    # Rescale to linearly interpolated norm
    target_norm = (1 - t) * norm_v0 + t * norm_v1
    return slerp_result * target_norm


def construct_filepath(model_name: str, shared_context: str, interpolation_layer: int, token_pair: List[str], n_steps: int, freeze_suffix: str = "") -> str:
    """Construct full filepath for activation file.

    Args:
        freeze_suffix: Optional freeze suffix like "_freeze_attn" or "_freeze_mlp"
    """
    context_clean = shared_context.replace(" ", "_").replace(".", "").replace(",", "").replace("'", "").replace('"', "")
    tokens_str = "_".join(token_pair)
    filename = f"interpolate_layer{interpolation_layer}{freeze_suffix}_{context_clean}_[{tokens_str}]_{n_steps}steps.pt"
    return f"./activations/{model_name}/{filename}"


def load_activations(model_name: str, shared_context: str, interpolation_layer: int, token_pair: List[str], n_steps: int, freeze_suffix: str = "") -> Dict:
    """Load activations from file.

    Args:
        freeze_suffix: Optional freeze suffix like "_freeze_attn" or "_freeze_mlp"
    """
    filepath = construct_filepath(model_name, shared_context, interpolation_layer, token_pair, n_steps, freeze_suffix)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Activations file not found: {filepath}")
    return torch.load(filepath, map_location='cpu')


def load_config(config_path: str = "./config.yaml") -> Dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_n_layers(activations: Dict) -> int:
    """Get number of layers from activation dictionary keys."""
    max_layer = max([int(k.split('_')[0].replace('layer', ''))
                     for k in activations.keys() if k.startswith('layer')])
    return max_layer + 1


def compute_relative_distances(activations: torch.Tensor) -> List[float]:
    """Compute relative distances from endpoint A to endpoint B.

    Args:
        activations: Tensor of shape [n_steps, ...] where ... can be any dimensions

    Returns:
        List of relative distances for each step
    """
    # Flatten to [n_steps, -1]
    activations_flat = activations.view(activations.shape[0], -1)

    endpoint_a = activations_flat[0]
    endpoint_b = activations_flat[-1]

    relative_distances = []
    for step_idx in range(activations_flat.shape[0]):
        current = activations_flat[step_idx]
        dist_to_a = torch.norm(current - endpoint_a).item()
        dist_to_b = torch.norm(current - endpoint_b).item()
        relative_distances.append(dist_to_a / (dist_to_a + dist_to_b))

    return relative_distances


def generate_interpolation_results_plot(
    data_dict: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
    suptitle: str,
    ylabel: str,
    output_path: str,
    n_steps: int,
    shared_context: str,
    token_pairs: List[List[str]],
    alpha_range: List[float] = [0, 1],
    skip_interpolation_layer: bool = True
) -> None:
    """
    Generate standardized interpolation results plot with two subplots.

    Args:
        data_dict: Dictionary with token pair keys (e.g., "very_quite", "very_white") mapping to either:
                   - Dict with layer keys (for multi-layer plots)
                   - Single tensor (for single-line plots)
        suptitle: Overall title for the plot
        ylabel: Y-axis label
        output_path: Path to save the plot
        n_steps: Number of interpolation steps
        shared_context: Shared context string (e.g., "The house at the end of the street was")
        token_pairs: List of token pairs (e.g., [["very", "quite"], ["very", "white"]])
        alpha_range: Range for alpha values (default: [0, 1])
        skip_interpolation_layer: Whether to skip layer 0 (interpolation layer) in multi-layer plots
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    y_min, y_max = float('inf'), float('-inf')
    has_multi_layer_data = False

    for i, token_pair in enumerate(token_pairs):
        ax = axes[i]
        token_key = f"{token_pair[0]}_{token_pair[1]}"
        data = data_dict[token_key]

        is_multi_layer = isinstance(data, dict)

        if is_multi_layer:
            # Filter layers, skipping layer 0 if requested
            valid_layers = [(k, v) for k, v in data.items()
                           if not (skip_interpolation_layer and 'Layer 0' in k)]

            has_multi_layer_data = True

            # Include every 5th layer in legend
            legend_indices = set(range(0, len(valid_layers), 5))

            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_layers)))

            for j, (layer_key, layer_data) in enumerate(valid_layers):
                # Extract layer number from key (e.g., "Layer 5" -> 5)
                label = None
                if j in legend_indices:
                    if 'Layer' in layer_key:
                        label = f"Layer {layer_key.split()[1]}"
                    else:
                        label = layer_key

                ax.plot(alphas, layer_data, alpha=0.8, color=colors[j], linewidth=1, label=label)
                y_min = min(y_min, layer_data.min().item())
                y_max = max(y_max, layer_data.max().item())
        else:
            # Single-line plot
            ax.plot(alphas, data, color='#2C2C2C', linewidth=2, alpha=0.9)
            y_min = min(y_min, data.min().item())
            y_max = max(y_max, data.max().item())

        # Reference lines and labels
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.text(0, -0.06, f'"{token_pair[0]}"', ha='center', va='top',
               fontsize=10, color='gray', alpha=0.9, transform=ax.get_xaxis_transform())
        ax.text(1, -0.06, f'"{token_pair[1]}"', ha='center', va='top',
               fontsize=10, color='gray', alpha=0.9, transform=ax.get_xaxis_transform())

        # Formatting
        ax.set_title(f'{shared_context} [{token_pair[0]} → {token_pair[1]}]', fontsize=14)
        ax.set_xlabel('Interpolation α', fontsize=12)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    # Shared y-limits
    y_padding = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Legend for multi-layer plots
    if has_multi_layer_data:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc='upper left', fontsize=9)

    plt.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

