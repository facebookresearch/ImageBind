#!/usr/bin/env python3
import os
import torch
from imagebind.models.imagebind_model import ImageBindModel, ModalityType


class ModularImageBind(ImageBindModel):
    """
    An extension of ImageBindModel that allows loading specific modalities only.
    """

    def __init__(
        self, modalities=None, weights_dir=".checkpoints/modality_specific", **kwargs
    ):
        """
        Initialize a modality-specific ImageBind model.

        Args:
            modalities: List of modalities to load (default: all modalities)
            weights_dir: Directory containing modality-specific weights
            **kwargs: Additional arguments to pass to ImageBindModel
        """
        # Initialize with all modalities to create the architecture
        super().__init__(**kwargs)

        # If no modalities specified, use all available
        if modalities is None:
            modalities = [
                ModalityType.VISION,
                ModalityType.TEXT,
                ModalityType.AUDIO,
                ModalityType.DEPTH,
                ModalityType.THERMAL,
                ModalityType.IMU,
            ]

        self.active_modalities = set(modalities)

        # Load weights for each modality
        for modality in modalities:
            self._load_modality_weights(modality, weights_dir)

    def _load_modality_weights(self, modality, weights_dir):
        """
        Load weights for a specific modality.

        Args:
            modality: Modality to load
            weights_dir: Directory containing modality-specific weights
        """
        weight_path = os.path.join(weights_dir, f"imagebind_{modality}.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Weights for {modality} not found at {weight_path}"
            )

        # Load modality-specific weights
        modality_state_dict = torch.load(weight_path, weights_only=True)

        # Create a temporary state dict for the current model state
        current_state_dict = self.state_dict()

        # Update only the parameters for this modality
        for k, v in modality_state_dict.items():
            if k in current_state_dict:
                current_state_dict[k] = v

        # Load the updated state dict
        self.load_state_dict(current_state_dict, strict=False)

        print(f"Loaded weights for {modality} modality")

    def forward(self, inputs):
        """
        Forward pass for the model, using only active modalities.

        Args:
            inputs: Dictionary of inputs for different modalities

        Returns:
            Dictionary of outputs for the active modalities
        """
        # Filter inputs to only use active modalities
        filtered_inputs = {
            k: v for k, v in inputs.items() if k in self.active_modalities
        }

        # Call the parent's forward method with filtered inputs
        return super().forward(filtered_inputs)


def load_modular_imagebind_huge(
    modalities=None, weights_dir=".checkpoints/modality_specific"
):
    """
    Helper function to load a modular ImageBind model with specific modalities.

    Args:
        modalities: List of modalities to load (default: all modalities)
        weights_dir: Directory containing modality-specific weights

    Returns:
        ModularImageBind model with requested modalities
    """
    model = ModularImageBind(
        modalities=modalities,
        weights_dir=weights_dir,
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )
    return model


# Example usage:
if __name__ == "__main__":
    from imagebind import data

    text_list = ["A dog.", "A car", "A bird"]
    image_paths = [
        ".assets/dog_image.jpg",
        ".assets/car_image.jpg",
        ".assets/bird_image.jpg",
    ]
    audio_paths = [
        ".assets/dog_audio.wav",
        ".assets/car_audio.wav",
        ".assets/bird_audio.wav",
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Example 1: Load only vision and text modalities
    print("Loading Vision-Text model...")
    model_vision_text = load_modular_imagebind_huge(
        modalities=[ModalityType.VISION, ModalityType.TEXT]
    )
    model_vision_text.to(device)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }

    # Perform inference
    with torch.no_grad():
        embeddings = model_vision_text(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
        ),
    )

    del model_vision_text
    del inputs
    del embeddings

    # Example 2: Load only audio modality
    print("Loading Audio model...")
    model_audio = load_modular_imagebind_huge(modalities=[ModalityType.AUDIO])
    model_audio.to(device)

    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    # Perform inference
    with torch.no_grad():
        embeddings = model_audio(inputs)

    print(
        "Audio: ",
        torch.softmax(
            embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.AUDIO].T, dim=-1
        ),
    )

    del model_audio
    del inputs
    del embeddings

    # Example 3: Create a multimodal model with vision, text, and audio
    print("Loading Multimodal model...")
    model_multimodal = load_modular_imagebind_huge(
        modalities=[ModalityType.VISION, ModalityType.TEXT, ModalityType.AUDIO]
    )
    model_multimodal.to(device)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    # Perform inference
    with torch.no_grad():
        embeddings = model_multimodal(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
        ),
    )
    print(
        "Audio x Text: ",
        torch.softmax(
            embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1
        ),
    )
    print(
        "Vision x Audio: ",
        torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1
        ),
    )

    del model_multimodal
    del inputs
    del embeddings
