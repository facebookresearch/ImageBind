# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List, Optional
from cog import BasePredictor, Input, Path
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

MODALITY_TO_PREPROCESSING = {
    ModalityType.TEXT: data.load_and_transform_text,
    ModalityType.VISION: data.load_and_transform_vision_data,
    ModalityType.AUDIO: data.load_and_transform_audio_data,
}


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        self.model = model.to("cuda")

    def predict(
        self,
        input: Path = Input(
            description="file that you want to embed. Needs to be text, vision, or audio.",
            default=None,
        ),
        text_input: str = Input(
            description="text that you want to embed. Provide a string here instead of a text file to input if you'd like.",
            default=None,
        ),
        modality: str = Input(
            description="modality of the input you'd like to embed",
            choices=list(MODALITY_TO_PREPROCESSING.keys()),
            default=ModalityType.VISION,
        ),
    ) -> List[float]:
        """Infer a single embedding with the model"""

        if not input and not text_input:
            raise Exception(
                "Neither input nor text_input were provided! Provide one in order to generate an embedding"
            )

        modality_function = MODALITY_TO_PREPROCESSING[modality]

        if modality == "text":
            if input and text_input:
                raise Exception(
                    f"Input and text_input were both provided! Only provide one to generate an embedding.\nInput provided: {input}\nText Input provided: {text_input}"
                )
            if text_input:
                input = text_input
            else:
                with open(input, "r") as f:
                    text_input = f.readlines()
                input = text_input

        device = "cuda"
        model_input = {modality: modality_function([input], device)}

        with torch.no_grad():
            embeddings = self.model(model_input)
        # print(type(embeddings))
        emb = embeddings[modality]
        return emb.cpu().squeeze().tolist()
