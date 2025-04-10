#!/usr/bin/env python3
import os
from memory_profiler import profile
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from functools import partial

import torch
import torch.nn as nn

from imagebind.models.helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from imagebind.models.multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)
from imagebind.models.transformer import MultiheadAttention, SimpleTransformer


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

        # Initialize with all modalities to create the architecture
        super().__init__(**kwargs)

        # Load weights for each modality
        for modality in modalities:
            self._load_modality_weights(modality, weights_dir)

    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
    ):
        if ModalityType.VISION in self.active_modalities:
            rgbt_stem = PatchEmbedGeneric(
                proj_stem=[
                    PadIm2Video(pad_type="repeat", ntimes=2),
                    nn.Conv3d(
                        in_channels=3,
                        kernel_size=kernel_size,
                        out_channels=vision_embed_dim,
                        stride=kernel_size,
                        bias=False,
                    ),
                ]
            )
            rgbt_preprocessor = RGBDTPreprocessor(
                img_size=[3, video_frames, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                rgbt_stem=rgbt_stem,
                depth_stem=None,
            )

        if ModalityType.TEXT in self.active_modalities:
            text_preprocessor = TextPreprocessor(
                context_length=77,
                vocab_size=49408,
                embed_dim=text_embed_dim,
                causal_masking=True,
            )

        if ModalityType.AUDIO in self.active_modalities:
            audio_stem = PatchEmbedGeneric(
                proj_stem=[
                    nn.Conv2d(
                        in_channels=1,
                        kernel_size=audio_kernel_size,
                        stride=audio_stride,
                        out_channels=audio_embed_dim,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
            )
            audio_preprocessor = AudioPreprocessor(
                img_size=[1, audio_num_mel_bins, audio_target_len],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                audio_stem=audio_stem,
            )

        if ModalityType.DEPTH in self.active_modalities:
            depth_stem = PatchEmbedGeneric(
                [
                    nn.Conv2d(
                        kernel_size=depth_kernel_size,
                        in_channels=1,
                        out_channels=depth_embed_dim,
                        stride=depth_kernel_size,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
            )

            depth_preprocessor = RGBDTPreprocessor(
                img_size=[1, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                rgbt_stem=None,
                depth_stem=depth_stem,
            )

        if ModalityType.THERMAL in self.active_modalities:
            thermal_stem = PatchEmbedGeneric(
                [
                    nn.Conv2d(
                        kernel_size=thermal_kernel_size,
                        in_channels=1,
                        out_channels=thermal_embed_dim,
                        stride=thermal_kernel_size,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
            )
            thermal_preprocessor = ThermalPreprocessor(
                img_size=[1, 224, 224],
                num_cls_tokens=1,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                thermal_stem=thermal_stem,
            )

        if ModalityType.IMU in self.active_modalities:
            imu_stem = PatchEmbedGeneric(
                [
                    nn.Linear(
                        in_features=48,
                        out_features=imu_embed_dim,
                        bias=False,
                    ),
                ],
                norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
            )

            imu_preprocessor = IMUPreprocessor(
                img_size=[6, 2000],
                num_cls_tokens=1,
                kernel_size=8,
                embed_dim=imu_embed_dim,
                pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
                imu_stem=imu_stem,
            )

        modality_preprocessors = {}
        if ModalityType.VISION in self.active_modalities:
            modality_preprocessors[ModalityType.VISION] = rgbt_preprocessor
        if ModalityType.TEXT in self.active_modalities:
            modality_preprocessors[ModalityType.TEXT] = text_preprocessor
        if ModalityType.AUDIO in self.active_modalities:
            modality_preprocessors[ModalityType.AUDIO] = audio_preprocessor
        if ModalityType.DEPTH in self.active_modalities:
            modality_preprocessors[ModalityType.DEPTH] = depth_preprocessor
        if ModalityType.THERMAL in self.active_modalities:
            modality_preprocessors[ModalityType.THERMAL] = thermal_preprocessor
        if ModalityType.IMU in self.active_modalities:
            modality_preprocessors[ModalityType.IMU] = imu_preprocessor

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    (
                        nn.LayerNorm(embed_dim, eps=1e-6)
                        if pre_transformer_ln
                        else nn.Identity()
                    ),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}

        if ModalityType.VISION in self.active_modalities:
            modality_trunks[ModalityType.VISION] = instantiate_trunk(
                vision_embed_dim,
                vision_num_blocks,
                vision_num_heads,
                pre_transformer_ln=True,
                add_bias_kv=False,
                drop_path=0.0,
            )
        if ModalityType.TEXT in self.active_modalities:
            modality_trunks[ModalityType.TEXT] = instantiate_trunk(
                text_embed_dim,
                text_num_blocks,
                text_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=False,
                drop_path=0.0,
            )
        if ModalityType.AUDIO in self.active_modalities:
            modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
                audio_embed_dim,
                audio_num_blocks,
                audio_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=audio_drop_path,
            )
        if ModalityType.DEPTH in self.active_modalities:
            modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
                depth_embed_dim,
                depth_num_blocks,
                depth_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=depth_drop_path,
            )
        if ModalityType.THERMAL in self.active_modalities:
            modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
                thermal_embed_dim,
                thermal_num_blocks,
                thermal_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=thermal_drop_path,
            )
        if ModalityType.IMU in self.active_modalities:
            modality_trunks[ModalityType.IMU] = instantiate_trunk(
                imu_embed_dim,
                imu_num_blocks,
                imu_num_heads,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=imu_drop_path,
            )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        if ModalityType.VISION in self.active_modalities:
            modality_heads[ModalityType.VISION] = nn.Sequential(
                nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
            )

        if ModalityType.TEXT in self.active_modalities:
            modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
                proj=nn.Sequential(
                    nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                    nn.Linear(text_embed_dim, out_embed_dim, bias=False),
                )
            )

        if ModalityType.AUDIO in self.active_modalities:
            modality_heads[ModalityType.AUDIO] = nn.Sequential(
                nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
            )

        if ModalityType.DEPTH in self.active_modalities:
            modality_heads[ModalityType.DEPTH] = nn.Sequential(
                nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
            )

        if ModalityType.THERMAL in self.active_modalities:
            modality_heads[ModalityType.THERMAL] = nn.Sequential(
                nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
            )

        if ModalityType.IMU in self.active_modalities:
            modality_heads[ModalityType.IMU] = nn.Sequential(
                nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
                SelectElement(index=0),
                nn.Dropout(p=0.5),
                nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
            )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        if ModalityType.VISION in self.active_modalities:
            modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)

        if ModalityType.TEXT in self.active_modalities:
            modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
                Normalize(dim=-1), LearnableLogitScaling(learnable=True)
            )

        if ModalityType.AUDIO in self.active_modalities:
            modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
            )

        if ModalityType.DEPTH in self.active_modalities:
            modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
            )

        if ModalityType.THERMAL in self.active_modalities:
            modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
            )

        if ModalityType.IMU in self.active_modalities:
            modality_postprocessors[ModalityType.IMU] = nn.Sequential(
                Normalize(dim=-1),
                LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
            )

        return nn.ModuleDict(modality_postprocessors)

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


@profile
def vision_text_example():
    from imagebind import data

    text_list = ["A dog.", "A car", "A bird"]
    image_paths = [
        ".assets/dog_image.jpg",
        ".assets/car_image.jpg",
        ".assets/bird_image.jpg",
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


@profile
def audio_example():
    from imagebind import data

    audio_paths = [
        ".assets/dog_audio.wav",
        ".assets/car_audio.wav",
        ".assets/bird_audio.wav",
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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


@profile
def multimodal_example():
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


@profile
def audio_thermal_example():
    from imagebind import data

    audio_paths = [
        ".assets/dog_audio.wav",
        ".assets/car_audio.wav",
        ".assets/bird_audio.wav",
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Example 4: Create a multimodal model with audio and thermal
    print("Loading Audio-Thermal model...")
    model_audio_thermal = load_modular_imagebind_huge(
        modalities=[ModalityType.AUDIO, ModalityType.THERMAL]
    )
    model_audio_thermal.to(device)

    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    # Perform inference
    with torch.no_grad():
        embeddings = model_audio_thermal(inputs)

    print(
        "Audio x Thermal: ",
        torch.softmax(
            embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.AUDIO].T, dim=-1
        ),
    )


if __name__ == "__main__":
    # vision_text_example()
    # audio_example()
    multimodal_example()
    # audio_thermal_example()
