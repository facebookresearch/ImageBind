# python c:/.../ImageBind/ranker.py -d ImageBind/.assets -i ImageBind/.assets/bird_audio.wav -n 3 > output.txt

import argparse, os, pathlib
import torch
from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
import ImageBind.data as data

args = argparse.ArgumentParser()
args.add_argument("-d") # Directory of the embedded dataset
args.add_argument("-i") # Item to find the similarities with
args.add_argument("-m") # Model path
args.add_argument("-n") # Top N items to return
args = args.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

modalities = {
    ".wav" : ModalityType.AUDIO,
    ".mp3" : ModalityType.AUDIO,
    ".png" : ModalityType.VISION,
    ".jpeg" : ModalityType.VISION,
    ".jpg" : ModalityType.VISION,
    ".mp4" : ModalityType.VISION,
    ".txt": ModalityType.TEXT
}

preprocess_fn = {
    ModalityType.AUDIO: data.load_and_transform_audio_data,
    ModalityType.VISION: data.load_and_transform_vision_data,
    ModalityType.TEXT: data.load_and_transform_text,
}

inputs = {
    ModalityType.AUDIO: [],
    ModalityType.VISION: [],
    ModalityType.TEXT: [],
}

files = inputs.copy()

for item in os.listdir(args.d):
    path = pathlib.PureWindowsPath(os.path.join(args.d,item))
    modality = modalities[path.suffix]

    if modality == ModalityType.TEXT:
        with open(item, "r") as f:
            texts = f.readlines()
            texts = [t.removesuffix("\n") for t in texts]
            texts = " ".join(texts)
            if texts != "":
                files[modality].append(path)
                inputs[modality].append(texts)
    else:
        inputs[modality].append(os.path.join(args.d,item))
        
keys = list(inputs.keys())
for modality in keys:
    if inputs[modality] == []:
        inputs.pop(modality)

keys = list(inputs.keys())
inputs = {modality: preprocess_fn[modality](inputs[modality], device) for modality in inputs.keys()}

model = torch.load(args.m) if args.m is not None else imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with torch.no_grad():
    embeddings = model(inputs)

main_input_path = pathlib.PureWindowsPath(args.i)
main_input_modality = modalities[main_input_path.suffix]
main_input = {main_input_modality: preprocess_fn[main_input_modality]([main_input_path], device)}

with torch.no_grad():
    main_embedding = model(main_input)

sim_matrices = {modality: torch.softmax(main_embedding[main_input_modality] @ embeddings[modality].T, dim=-1) for modality in inputs.keys()}

scores={}
for modality in inputs.keys():
    for i in range(len(inputs[modality])):
        score = sim_matrices[modality][0][i]
        file = files[modality][i]
        scores[file] = score

def get_top_files(dictionary, N):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:N]
    top_strings = [(item[0], f"{item[1]*100} %") for item in top_items]
    return top_strings

print(*get_top_files(scores, int(args.n) if args.n is not None else 5), sep="\n")
