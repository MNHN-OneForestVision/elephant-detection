import os
import re
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import faiss
from faiss import IndexFlatL2
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor


class ViTEmbedder(nn.Module):
    """
    A neural network module for embedding images using a Vision Transformer (ViT).

    This class utilizes a pre-trained ViT model to generate embeddings from input images.
    The embeddings are extracted from the pooler output of the ViT model.

    Attributes:
        vit (ViTModel): The pre-trained Vision Transformer model.
        hidden_size (int): The size of the hidden layer in the ViT model.

    Args:
        model_name (str): The name of the pre-trained ViT model to use. Defaults to "google/vit-base-patch16-224-in21k".

    Methods:
        forward(x): Computes the embeddings for the input images.
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        Initializes the ViTEmbedder with a specified pre-trained ViT model.

        :param model_name: The name of the pre-trained ViT model to use. 
                           It should be a valid model identifier from the Hugging Face model hub.
                           Defaults to "google/vit-base-patch16-224-in21k".
        """
        super().__init__()
        logging.info(f'Initializing ViTEmbedder with model: {model_name}')
        try:
            self.vit = ViTModel.from_pretrained(model_name)
            self.hidden_size = self.vit.config.hidden_size
            logging.info('Model loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading model {model_name}: {e}')
            raise

    def forward(self, x) :
        """
        Computes the embeddings for the input images using the Vision Transformer model.

        :param x: A tensor representing the pixel values of the input images.
        :return: A tensor containing the embeddings extracted from the pooler output of the ViT model.
        """
        outputs = self.vit(pixel_values=x)
        return outputs.pooler_output


class TripletEarDataset(Dataset):
    """
    Dataset class for the Triplet Ear-Dataset.
    """
    def __init__(self, image_folder, labels_dict, transform=None):
        self.image_folder = image_folder
        self.labels_dict = labels_dict
        self.transform = transform
        self.id_to_images = self._group_by_id()

    def _group_by_id(self) -> dict:
        id_to_imgs = {}
        for fname, label in self.labels_dict.items():
            id_to_imgs.setdefault(label, []).append(fname)
        return id_to_imgs

    def __len__(self) -> int:
        return len(self.labels_dict)

    def __getitem__(self, idx) -> tuple[any, any, any]:
        anchor_fname = list(self.labels_dict.keys())[idx]
        anchor_id = self.labels_dict[anchor_fname]

        positive_fname = anchor_fname
        while positive_fname == anchor_fname:
            positive_fname = np.random.choice(self.id_to_images[anchor_id])

        negative_id = anchor_id
        while negative_id == anchor_id:
            negative_id = np.random.choice(list(self.id_to_images.keys()))
        negative_fname = np.random.choice(self.id_to_images[negative_id])

        def load_image(fname):
            img_path = os.path.join(self.image_folder, fname)
            img = Image.open(img_path)
            return self.transform(img) if self.transform else img

        return load_image(anchor_fname), load_image(positive_fname), load_image(negative_fname)


def build_labels_dict(dataset_path) -> dict:
    """
    Groups image filenames by their associated labels.

    :return: A dictionary where keys are labels and values are lists of filenames corresponding to each label.
    """
    labels = {}
    # pattern = re.compile(r'^(\d+)_\d+_crop_\d+\.jpg$')
    pattern = re.compile(r'.*[lr]-(\d+).*')
    for fname in os.listdir(dataset_path):
        match = pattern.match(fname)
        if match:
            elephant_id = match.group(1)
            labels[fname] = elephant_id
    return labels


def train_model(dataset, model, save_dir='checkpoints', epochs=600, batch_size=1, lr=1e-5, margin=1.0) -> None:
    """
    Train a ViT model using triplet margin loss and save the best and the last models.

    :param dataset: A Pytorch Dataset object containing the images and labels.
    :param model: A Pytorch ViT model.
    :param save_dir: Directory to save the trained model.
    :param epochs: Number of epochs to train the model.
    :param batch_size: Number of images, samples per batch.
    :param lr: Learning rate for the optimizer.
    :param margin: Margin for triplet loss function.
    :return: None
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for anchor, positive, negative in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logging.info(f'Best model saved at {save_dir}.')
        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    logging.info(f'Last model saved at {save_dir}.')


def extract_embeddings(model, image_folder, labels_dict, output_faiss='faiss.index', output_npy='embeddings.npy') -> tuple[IndexFlatL2 | IndexFlatL2 | IndexFlatL2, list[any]]:
    """
    Extract embedding from images using ViT model.

    :param model: A Pytorch ViT model.
    :param image_folder: Path to the folder containing the images.
    :param labels_dict: Dictionary containing the labels of the images.
    :param output_faiss: Filename of the faiss index output file.
    :param output_npy: Filename of the embeddings output file.
    :return: A tuple containing the embeddings extracted from the pooler output of the ViT model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    def preprocess(img):
        return processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

    embeddings = []
    filenames = []

    with torch.no_grad():
        for fname in tqdm(labels_dict.keys(), desc='Extracting embeddings'):
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path)
            tensor = preprocess(img).unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy()
            embeddings.append(emb)
            filenames.append(fname)

    embeddings = np.vstack(embeddings)
    np.save(output_npy, embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, output_faiss)

    logging.info(f'Indexed {len(filenames)} embeddings.')
    return index, filenames

def main() -> None:
    image_folder = '/path/to/the/dataset/'
    labels_dict = build_labels_dict(image_folder)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    transform = lambda img: processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

    dataset = TripletEarDataset(image_folder, labels_dict, transform=transform)
    model = ViTEmbedder()

    train_model(dataset, model, save_dir='checkpoints_v2', epochs=600)

    # model.load_state_dict(torch.load('checkpoints_v2/best_model.pth'))
    # extract_embeddings(model, image_folder, labels_dict)


if __name__ == '__main__':
    main()