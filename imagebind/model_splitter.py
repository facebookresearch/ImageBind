#!/usr/bin/env python3
import os
import torch
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from collections import OrderedDict


def split_imagebind_model(pretrained=True, save_dir=".checkpoints/modality_specific"):
    """
    Load the full ImageBind model, split it by modality, and save modality-specific weights.

    Args:
        pretrained: Whether to load pretrained weights
        save_dir: Directory to save modality-specific weights
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the full model with pretrained weights
    print(f"Loading full ImageBind model with pretrained={pretrained}...")
    full_model = imagebind_huge(pretrained=pretrained)

    # Define the modalities we want to split
    modalities = [
        ModalityType.VISION,
        ModalityType.TEXT,
        ModalityType.AUDIO,
        ModalityType.DEPTH,
        ModalityType.THERMAL,
        ModalityType.IMU,
    ]

    for modality in modalities:
        print(f"Creating weights for {modality} modality...")

        # Create a dictionary for modality-specific state dict
        modality_state_dict = OrderedDict()

        # Get the full state dict
        full_state_dict = full_model.state_dict()

        # Extract common parameters (not specific to any modality)
        common_prefixes = []

        # Extract modality-specific parameters
        modality_prefixes = [
            f"modality_preprocessors.{modality}",
            f"modality_trunks.{modality}",
            f"modality_heads.{modality}",
            f"modality_postprocessors.{modality}",
        ]

        # Collect all parameters for this modality
        for k, v in full_state_dict.items():
            # Check if this is a modality-specific parameter
            is_modality_specific = any(
                k.startswith(prefix) for prefix in modality_prefixes
            )
            is_common = any(k.startswith(prefix) for prefix in common_prefixes)

            if is_modality_specific or is_common:
                modality_state_dict[k] = v

        # Save modality-specific state dict
        save_path = os.path.join(save_dir, f"imagebind_{modality}.pth")
        torch.save(modality_state_dict, save_path)
        print(f"Saved {modality} weights to {save_path}")
        print(f"Number of parameters: {len(modality_state_dict)}")

    print("Finished splitting model.")


if __name__ == "__main__":
    split_imagebind_model(pretrained=True)
