import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import os
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
from utils.getter import *
import numpy as np


def load_image_tensor(image_path, device):
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = F.interpolate(image_tensor, size=224)
    print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor.to(device)


def compute_similar_images(model, image_path, num_images, embedding, device):
    image_tensor = load_image_tensor(image_path, device)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = model.get_embedding(
            image_tensor).cpu().detach().numpy()

    print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape(
        (image_embedding.shape[0], -1))
    print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    print(indices_list)
    return indices_list


def plot_similar_images(indices_list):
    indices = indices_list[0]
    for index in indices:
        img_name = str(index) + ".jpg"
        img_path = os.path.join('test_data/' + img_name)
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    TEST_IMAGE_PATH = "./test_data/98.jpg"
    NUM_IMAGES = 5
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    model = TripletNet(ResNetExtractor(version=152))
    model.load_state_dict(torch.load(
        hp.ckp, map_location=device)['state_dict'])
    model.eval()
    model.to(device)

    # Loads the embedding
    embedding = np.load(hp.embed_path)

    indices_list = compute_similar_images(
        TEST_IMAGE_PATH, NUM_IMAGES, embedding, device)
    plot_similar_images(indices_list)
