# Model Card for ImageBind

Multimodal joint embedding model for image/video, text, audio, depth, IMU, and thermal images.
Input any of the six modalities and get the same sized embedding that can be used for cross-modal and multimodal tasks.

# Model Details

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
Multimodal joint embedding model for image/video, text, audio, depth, IMU, and thermal images

- **Developed by:** Meta AI
- **Model type:** Multimodal model
- **Language(s) (NLP):** en
- **License:** CC BY-NC-SA 4.0
- **Resources for more information:**
    - [GitHub Repo](https://github.com/facebookresearch/ImageBind)


# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
This model is intended only for research purposes. It provides a joint embedding space for different modalities -- image/video, text, audio, depth, IMU and thermal images.
We hope that these joint embeddings can be used for a variety of different cross-modal research, e.g., cross-modal retrieval and combining embeddings from different modalities.

## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

This model is *NOT* intended to be used in any real world application -- commercial or otherwise.
It may produce harmful associations with different inputs.
The model needs to be investigated and likely re-trained on specific data for any such application.
The model is expected to work better on web-based visual data since it was trained on such data.
The text encoder is likely to work only on English language text because of the underlying training datasets.

# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
Open-domain joint embedding models are prone to producing specific biases, e.g., study from [CLIP](https://github.com/openai/CLIP/blob/main/model-card.md#bias-and-fairness).
Since our model uses such models as initialization, it will exhibit such biases too.
Moreover, for learning joint embeddings for other modalities such as audio, thermal, depth, and IMU we leverage datasets that are relatively small. These joint embeddings are thus limited to the concepts present in the datasets. For example, the thermal datasets we used are limited to outdoor street scenes, while the depth datasets are limited to indoor scenes.



# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

ImageBind uses image-paired data for training -- (image, X) where X is one of text, audio, depth, IMU or thermal data.
In particular, we initialize and freeze the image and text encoders using an OpenCLIP ViT-H encoder.
We train audio embeddings using Audioset, depth embeddings using the SUN RGB-D dataset, IMU using the Ego4D dataset and thermal embeddings using the LLVIP dataset.
We provide the exact training data details in the paper.


## Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
Please refer to the research paper and github repo for exact details on this.

# Evaluation

## Testing Data, Factors & Metrics

We evaluate the model on a variety of different classification benchmarks for each modality.
The evaluation details are presented in the paper.
The models performance is measured using standard classification metrics such as accuracy and mAP.

# Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```


# Model Card Contact

Please reach out to the authors at: rgirdhar@meta.com imisra@meta.com alaaelnouby@gmail.com

# How to Get Started with the Model

Our github repo provides a simple example to extract embeddings from images, audio etc.
