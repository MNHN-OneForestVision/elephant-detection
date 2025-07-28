from typing import AnyStr, LiteralString
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import faiss
import numpy as np
import os
import time
import logging

def timer(func):
    """
    Decorator that measures and prints the execution time of a function.

    :param func: The function whose execution time is to be measured.
    :return: The result of the function call.
    """
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        end: float = time.time()
        print(f'{func.__name__} took {end - start} seconds')
        return result
    return wrapper


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



def train(base_model, dataset: str = "african-wildlife.yaml", save_name:str = "my_yolo", epochs: int = 300, imgsz:int = 640):
    """
    Trains a YOLO model using the specified dataset and parameters.

    :param base_model: Path to the pre-trained YOLO model to use as a starting point (e.g., "yolov11n.pt").
    :param dataset: Path to the dataset configuration file in YOLO format (default is "african-wildlife.yaml").
    :param save_name: Name to save the trained model under (default is "my_yolo").
    :param epochs: Number of training epochs (default is 300).
    :param imgsz: Image size to be used during training (default is 640).
    :return: None
    """
    model = YOLO(base_model)
    model.train(data=dataset, epochs=epochs, imgsz=imgsz)
    model.save(save_name)


def analyze(model_name: str, video_path: str, conf: float = 0.75, save_analyze: bool = False, graph_name: str = None, show: bool = False):
    """
    Analyzes a video using a specified YOLO model and optionally saves the analysis results.

    This function processes a video to detect objects using a YOLO model. It can save the
    detection results and generate a graph of object counts over time.

    :param model_name: The name of the YOLO model to use for analysis.
    :param video_path: The path to the video file to be analyzed.
    :param conf: The confidence threshold for object detection. Defaults to 0.75.
    :param save_analyze: Whether to save the analysis results. Defaults to False.
    :param graph_name: The filename to save the generated graph. If None, no graph is saved.
    :param show: Whether to show the video. Defaults to False.
    :return: The results of the video analysis.
    """
    model = YOLO(model_name)
    results = model(video_path, stream=True, save=save_analyze, conf=conf, show=show)
    if graph_name is not None:
        create_graph(results, graph_name, model)
    return results


def create_graph(results: list, graph_name: str, model):
    counts = defaultdict(list)
    frame_numbers = []
    for frame_id, result in enumerate(results):
        current_counts = defaultdict(int)
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            current_counts[class_name] += 1

        for class_name in model.names.values():
            counts[class_name].append(current_counts[class_name])

        frame_numbers.append(frame_id)

    plt.figure(figsize=(12, 6))
    for class_name, class_counts in counts.items():
        if any(class_counts):
            plt.plot(frame_numbers, class_counts, label=class_name)

    plt.xlabel("N° Frame")
    plt.ylabel("Nbr d'Éléphant")
    plt.title("Beta")
    plt.legend()
    plt.savefig(graph_name)

def train_segmentation():
    model = YOLO("yolo11n-seg.pt")

    result = model.train(
        data="/home/avalanche/MNHN/mnhm-yolo-elephant-head-recognizer/mnhm_seg.yaml",
        epochs=1200,
        imgsz=640,
    )
    model.save("v001_11seg.pt")

    model8 = YOLO("yolov8n-seg.pt")

    result = model8.train(
        data="/home/avalanche/MNHN/mnhm-yolo-elephant-head-recognizer/mnhm_seg.yaml",
        epochs=1200,
        imgsz=640,
    )
    model8.save("v001_8seg.pt")


class ViTEmbedder(nn.Module):
    """
    Vision Transformer (ViT) Embedder class for extracting embeddings from images.

    Attributes:
        vit (ViTModel): The Vision Transformer model used for embedding extraction.
        hidden_size (int): The size of the hidden layer in the ViT model.

    Args:
        model_name (str): The name of the pre-trained ViT model to load. Defaults to "google/vit-base-patch16-224-in21k".

    Methods:
        forward(x): Processes input image tensor `x` and returns the pooled output embeddings.
    """
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.vit.config.hidden_size

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.pooler_output


def reset_database(faiss_index_path="faiss.index", embeddings_path="embeddings.npy", filenames_path="filenames.npy", dim=768) -> None:
    """
    Resets the faiss index and embeddings.

    :param faiss_index_path:
    :param embeddings_path:
    :param filenames_path:
    :param dim:
    :return:
    """
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, faiss_index_path)

    np.save(embeddings_path, np.zeros((0, dim), dtype=np.float32))
    np.save(filenames_path, [])


def safe_load_npy(path, shape_if_empty=None, dtype=np.float32) -> np.ndarray:
    """
    Safely loads a NumPy array from a file, with handling for empty or non-existent files.

    :param path: The file path to load the NumPy array from.
    :param shape_if_empty: Optional shape for the array if the file is empty or non-existent.
    :param dtype: Data type of the array to return if the file is empty or non-existent. Defaults to np.float32.
    :return: The loaded NumPy array, or an empty/zero-filled array if the file is empty or non-existent.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        if shape_if_empty is None:
            return np.array([], dtype=dtype)
        else:
            return np.zeros(shape_if_empty, dtype=dtype)
    else:
        try:
            return np.load(path)
        except Exception as e:
            logging.error(f'Try to read {path}. {e}')
            if shape_if_empty is None:
                return np.array([], dtype=dtype)
            else:
                return np.zeros(shape_if_empty, dtype=dtype)

def get_all_file_paths(folder: AnyStr) -> list[LiteralString]:
    paths = []
    for root, dirs, files in os.walk(folder):
        # for f in os.listdir(folder_images)
        for file in files:
            paths.append(os.path.join(root, file))
    print(len(paths))
    return paths


def erase_label_uuid(fp):
    paths = get_all_file_paths(fp)
    for path in paths:
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        if len(filename) > 9:
            new_filename = filename[9:]
            new_path = os.path.join(dirname, new_filename)
            if not os.path.exists(new_path):
                os.rename(path, new_path)
                print(f"Renommé : {filename} -> {new_filename}")
            else:
                print(f"ATTENTION : {new_filename} existe déjà, {filename} ignoré.")
        else:
            print(f"Nom trop court, ignoré : {filename}")


def lister_et_indexer_fichiers(dossier):
    fichiers = sorted(os.listdir(dossier))
    pattern = re.compile(r"([lr])-(\d+)")
    index_dict_l = defaultdict(list)
    index_dict_r = defaultdict(list)

    for f in fichiers:
        match = pattern.search(f)
        if match:
            cote = match.group(1)
            chiffre = match.group(2)
            if cote == 'l':
                index_dict_l[chiffre].append(f)
            elif cote == 'r':
                index_dict_r[chiffre].append(f)

    premiere_l = [sorted(values)[0] for values in index_dict_l.values() if values]
    premiere_r = [sorted(values)[0] for values in index_dict_r.values() if values]

    return (index_dict_l, premiere_l), (index_dict_r, premiere_r)
