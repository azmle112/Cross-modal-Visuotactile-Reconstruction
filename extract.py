"""特征提取"""

import os
import numpy
import torch
from CLIP.clip import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("./ViT-B-32.pt", device=device)

directory_path = '___LMT_TextureDB___/Training/ImageMagnified/'
features = []
for filename in os.listdir(directory_path):
    image = preprocess(Image.open(f"{directory_path}{filename}")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
    features.append(image_features.reshape(512))
numpy.save("./features.npy", numpy.array(features))

