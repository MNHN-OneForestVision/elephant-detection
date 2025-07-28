#!/usr/bin/python3
import os
import logging
import argparse
from pathlib import Path

import faiss
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO
from transformers import ViTImageProcessor

import lib
from lib import ViTEmbedder, timer, safe_load_npy


class EarRecognizerSystem:
    def __init__(self,
                 yolo_model_path: str = 'models/segmentation/seg_ears_lr_v2.pt',
                 vit_model_path: str = 'models/identification/vit_v2.pth',
                 faiss_index_left_path: str = 'db/index/faiss_left.index',
                 faiss_index_right_path: str = 'db/index/faiss_right.index',
                 embeddings_left_path: str = 'db/embeddings/embeddings_left.npy',
                 embeddings_right_path: str = 'db/embeddings/embeddings_right.npy',
                 filenames_left_path: str = 'db/path/filenames_left.npy',
                 filenames_right_path: str = 'db/path/filenames_right.npy',
                 original_paths_left_path: str = 'db/path/original_paths_left.npy',
                 original_paths_right_path: str = 'db/path/original_paths_right.npy',
                 vit_name: str = 'google/vit-base-patch16-224-in21k',
                 embedding_dim: int = 768):
        self.yolo_model_path = yolo_model_path
        self.vit_model_path = vit_model_path
        self.faiss_index_left_path = faiss_index_left_path
        self.faiss_index_right_path = faiss_index_right_path
        self.embeddings_left_path = embeddings_left_path
        self.embeddings_right_path = embeddings_right_path
        self.filenames_left_path = filenames_left_path
        self.filenames_right_path = filenames_right_path
        self.original_paths_left_path = original_paths_left_path
        self.original_paths_right_path = original_paths_right_path

        self.vit_name = vit_name
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.yolo_model = self._load_yolo_model()
        self.vit_model = self._load_vit_model()
        self.processor = ViTImageProcessor.from_pretrained(self.vit_name)

        self.faiss_left = self._safe_load_faiss_index(True)
        self.faiss_right = self._safe_load_faiss_index(False)
        self.embeddings_left = safe_load_npy(self.embeddings_left_path, shape_if_empty=(0, self.embedding_dim))
        self.embeddings_right = safe_load_npy(self.embeddings_right_path, shape_if_empty=(0, self.embedding_dim))
        self.left_filenames = safe_load_npy(self.filenames_left_path).tolist() if os.path.exists(
            self.filenames_left_path) else []
        self.right_filenames = safe_load_npy(self.filenames_right_path).tolist() if os.path.exists(
            self.filenames_right_path) else []

        self.left_original_paths = safe_load_npy(self.original_paths_left_path).tolist() if os.path.exists(
            self.original_paths_left_path) else []
        self.right_original_paths = safe_load_npy(self.original_paths_right_path).tolist() if os.path.exists(
            self.original_paths_right_path) else []

    def reset_index(self) -> None:
        """
        Resets the FAISS index and associated NPY files to their initial state.

        This method creates new FAISS indices for both left and right paths with
        the specified embedding dimension, and writes them to disk. It also resets
        the embeddings, filenames, and original paths NPY files to empty arrays.

        :return: None
        """
        logging.warning("Resetting FAISS index and NPY files...")

        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        faiss.write_index(faiss_index, self.faiss_index_left_path)
        faiss.write_index(faiss_index, self.faiss_index_right_path)

        embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        np.save(self.embeddings_left_path, embeddings)
        np.save(self.embeddings_right_path, embeddings)

        filenames = []
        np.save(self.filenames_left_path, filenames)
        np.save(self.filenames_right_path, filenames)

        np.save(self.original_paths_left_path, filenames)
        np.save(self.original_paths_right_path, filenames)
        logging.info("Reset complete: faiss.index, embeddings.npy, filenames.npy, original_paths.npy")

    def _load_yolo_model(self):
        """
        Loads the YOLO model specified by the class attribute.

        This method logs the loading process and initializes the YOLO model
        using the path provided in `self.yolo_model_path`.

        :return: The loaded YOLO model.
        """
        logging.info(f"Loading YOLO model from {self.yolo_model_path}")
        return YOLO(self.yolo_model_path)

    def _load_vit_model(self):
        """
        Loads the Vision Transformer (ViT) model specified by the class attributes.

        This method initializes a ViTEmbedder with the model name and loads its state
        from the specified path. The model is moved to the appropriate device and set
        to evaluation mode. If an error occurs during loading, it is logged and raised.

        :return: The loaded ViT model.
        :raises Exception: If there is an error loading the model.
        """
        logging.info(f"Loading ViT model {self.vit_name} from {self.vit_model_path}")
        try:
            model = ViTEmbedder(model_name=self.vit_name).to(self.device)
            model.load_state_dict(torch.load(self.vit_model_path, map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            logging.error(e)
            raise e

    def _safe_load_faiss_index(self, left) -> faiss.Index:
        """
        Safely loads a FAISS index from a specified path, creating a new index if the file does not exist.

        This method checks for the existence of a FAISS index file at the specified path
        (left or right based on the `left` parameter). If the file is not found, it logs a
        warning and creates a new FAISS index with the specified embedding dimension.

        :param left: A boolean indicating whether to load the left or right FAISS index.
        :return: A FAISS index object, either loaded from the file or newly created.
        """
        path = self.faiss_index_left_path if left else self.faiss_index_right_path
        if not os.path.exists(path):
            logging.warning(f"FAISS index not found at {path}. Creating a new one.")
            return faiss.IndexFlatL2(self.embedding_dim)
        return faiss.read_index(path)

    def create_crops(self, class_index, confs, boxes, img_array, save_cropped, output_dir, image_path):
        """
        Creates and optionally saves cropped images from detected bounding boxes.

        This method processes the input image array to extract crops based on the
        provided bounding boxes and class indices. It saves the crops to the specified
        output directory if `save_cropped` is True, ensuring unique filenames.

        :param class_index: List of indices indicating which bounding boxes to crop.
        :param confs: List of confidence scores for each detected object.
        :param boxes: List of bounding box coordinates for detected objects.
        :param img_array: NumPy array representation of the image to crop from.
        :param save_cropped: Boolean indicating whether to save the cropped images.
        :param output_dir: Directory path where cropped images will be saved.
        :param image_path: Path of the original image, used for naming crops.
        :return: List of tuples, each containing a PIL image of the crop and its save path.
        """
        crops = []
        for i, idx in enumerate(class_index):
            logging.info(f'confidences of {confs[i]}')
            x1, y1, x2, y2 = map(int, boxes[idx])
            crop = img_array[y1:y2, x1:x2]
            pil_crop = Image.fromarray(crop)

            os.makedirs(output_dir, exist_ok=True)
            base_filename = Path(image_path).stem
            index = 1
            while True:
                filename = f"{base_filename}_{index}.jpg"
                save_path = os.path.join(output_dir, filename)
                if not os.path.exists(save_path):
                    break
                index += 1

            crops.append((pil_crop, save_path))

            if save_cropped:
                pil_crop.save(save_path)
                logging.info(f'Saved cropped image to {save_path}')

        return crops

    @timer
    def extract_ear_from_image(
            self,
            image_path: str,
            save_cropped: bool = False
    ) -> tuple[list, list]:
        """
        Extracts ear crops from an image using a YOLO model and optionally saves them.

        This method processes the input image to detect ear regions, extracts the
        corresponding crops, and returns them. It can also save the cropped images
        to specified directories for left and right ears.

        :param image_path: The path to the image file to process.
        :param save_cropped: If True, saves the cropped ear images to disk.
        :return: A tuple containing two lists of tuples, each with a PIL image and its save path.
                 The first list is for left ear crops, and the second is for right ear crops.
        """
        results = self.yolo_model(image_path, conf=0.80)[0]
        if not results.boxes or len(results.boxes.xyxy) == 0:
            logging.error(f'No boxes found for {image_path}')
            return [], []

        class_ids = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_1_indices = np.where(class_ids == 1)[0]
        class_0_indices = np.where(class_ids == 0)[0]

        pil_image = Image.open(image_path).convert('LA')
        pil_image = pil_image.convert("RGB")

        img_array = np.array(pil_image)
        output_dir_left = './db/images/left'
        output_dir_right = './db/images/right'

        return self.create_crops(class_0_indices, confidences, boxes, img_array, save_cropped, output_dir_left,
                                 image_path), self.create_crops(class_1_indices, confidences, boxes, img_array,
                                                                save_cropped, output_dir_right, image_path)

    @timer
    def compute_embedding(self, pil_image: Image) -> np.ndarray:
        """
        Computes the embedding of a given PIL image using a ViT model.

        This method processes the input image to generate a tensor suitable for
        the ViT model, computes the embedding, and returns it as a NumPy array.

        :param pil_image: The PIL image to compute the embedding for.
        :return: A NumPy array representing the computed embedding.
        """
        inputs = self.processor(pil_image, return_tensors='pt')['pixel_values'].to(self.device)
        with torch.no_grad():
            emb = self.vit_model(inputs).cpu().numpy()
        return emb

    @timer
    def search_similar(self, pil_image: Image, top_k: int = 3, left: bool = True):
        """
        Searches for images similar to the given PIL image in the FAISS index.

        This method computes the embedding of the provided image and searches
        for the top_k most similar images in the specified FAISS index (left or right).
        It returns a list of tuples containing the filename, distance, and original path
        of each similar image found.

        :param pil_image: The PIL image to search for similar images.
        :param top_k: The number of top similar images to retrieve.
        :param left: A boolean indicating whether to search in the left or right FAISS index.
        :return: A list of tuples with each tuple containing the filename, distance, and original path.
        """
        results: list = []

        faiss_index = self.faiss_left if left else self.faiss_right
        filenames = self.left_filenames if left else self.right_filenames
        original_paths = self.left_original_paths if left else self.right_original_paths

        if faiss_index.ntotal == 0:
            logging.error('No faiss index found')
            return results

        emb = self.compute_embedding(pil_image)
        if faiss_index.ntotal < top_k:
            top_k = faiss_index.ntotal
        D, I = faiss_index.search(emb, top_k)

        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if len(filenames) > idx > -1:
                filename = filenames[idx]
                original_path = original_paths[idx] if len(original_paths) > idx else "Unknown"
                logging.info(f'Rank {rank + 1}: {filename} from {original_path} dist {dist}')
                results.append((filename, dist, original_path))
            else:
                logging.error(f'Index {idx} out of bound in FAISS')
        return results

    @timer
    def add_to_faiss(
            self,
            pil_crop: Image,
            source_filename: str,
            original_image_path: str,
            left: bool) -> None:
        """
        Adds an image crop to the FAISS index and updates related data structures.

        This method computes the embedding of the given image crop and adds it to the
        appropriate FAISS index (left or right). It also updates the embeddings, filenames,
        and original paths arrays, saving them to disk.

        :param pil_crop: The PIL image crop to be added to the FAISS index.
        :param source_filename: The filename of the source image.
        :param original_image_path: The original path of the source image.
        :param left: A boolean indicating whether to add to the left or right FAISS index.
        :return: None
        """
        faiss_index = self.faiss_left if left else self.faiss_right
        embeddings = self.embeddings_left if left else self.embeddings_right
        filenames = self.left_filenames if left else self.right_filenames
        original_paths = self.left_original_paths if left else self.right_original_paths

        embeddings_path = self.embeddings_left_path if left else self.embeddings_right_path
        filenames_path = self.filenames_left_path if left else self.filenames_right_path
        original_paths_path = self.original_paths_left_path if left else self.original_paths_right_path
        faiss_index_path = self.faiss_index_left_path if left else self.faiss_index_right_path

        emb = self.compute_embedding(pil_crop)
        faiss_index.add(emb)
        embeddings = np.vstack([embeddings, emb])
        filenames.append(source_filename)
        original_paths.append(original_image_path)
        np.save(embeddings_path, embeddings)
        np.save(filenames_path, filenames)
        np.save(original_paths_path, original_paths)
        faiss.write_index(faiss_index, faiss_index_path)
        logging.info(f'Image add in FAISS : {source_filename} from {original_image_path}')

    def run(
            self,
            image_path: str,
            add: bool = False,
            extract: bool = False) -> None:
        """
        Executes the ear recognition process on the given image.

        This method extracts ear crops from the specified image, searches for similar
        images in the database, and optionally adds the crops to the FAISS index.
        It also displays and saves the search results.

        :param image_path: The path to the image file to process.
        :param add: If True, adds the extracted ear crops to the FAISS index.
        :param extract: If True, saves the cropped ear images.
        :return: None
        """
        logging.info(f'Start ear detection with option add: {add}, extract: {extract}.\n image_path: {image_path}')
        pil_crops_left, pil_crops_right = self.extract_ear_from_image(image_path, save_cropped=extract)
        if not pil_crops_left and not pil_crops_right:
            logging.critical(f'No ears found in {image_path}')
            return

        for i, (pil_crop, filename) in enumerate(pil_crops_left):
            results = self.search_similar(pil_crop, 15, True)

            if add:
                self.add_to_faiss(pil_crop, filename, image_path, True)
            show_results(pil_crop, results, Path(image_path).stem, './db/images/left', save_dir='./results/left')

        for i, (pil_crop, filename) in enumerate(pil_crops_right):
            results = self.search_similar(pil_crop, 15, False)

            if add:
                self.add_to_faiss(pil_crop, filename, image_path, False)
            show_results(pil_crop, results, Path(image_path).stem, './db/images/right', save_dir='./results/right')


@timer
def show_results(
    pil_query_image,
    result_filenames_with_scores,
    img_name,
    db_folder,
    save_dir='archive'
):
    """
    Displays and saves the results of an image query alongside its matches.

    This function visualizes the query image and its corresponding result images
    with their scores in a grid layout. It saves the visualization as a PNG file
    in the specified directory.

    :param pil_query_image: The query image in PIL format.
    :param result_filenames_with_scores: A list of tuples containing result filenames and their scores.
    :param img_name: The name of the query image.
    :param db_folder: The directory containing the database images.
    :param save_dir: The directory where the result image will be saved. Defaults to 'archive'.
    :return: None
    """
    os.makedirs(save_dir, exist_ok=True)
    existing_svgs = [f for f in os.listdir(save_dir) if f.endswith('.png') and f[:-4].isdigit()]
    next_index = max([int(f[:-4]) for f in existing_svgs], default=-1) + 1
    save_path = os.path.join(save_dir, f"{next_index}.png")

    n_results = len(result_filenames_with_scores)
    ncols = n_results + 1
    nrows = 2

    plt.figure(figsize=(ncols * 4, nrows * 4))
    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.10, wspace=0.15, hspace=0.25)

    # Image requête
    plt.subplot(nrows, ncols, 1)
    query = f'{img_name}'
    plt.title(query, fontsize=14, fontweight='bold', pad=10)
    plt.imshow(pil_query_image)
    plt.axis('off')
    for i, result_tuple in enumerate(result_filenames_with_scores):
        if len(result_tuple) == 2:
            fname, dist = result_tuple
            original_path = None
        else:
            fname, dist, original_path = result_tuple

        crop_subplot_idx = i + 2

        img_path = os.path.join(db_folder, fname)
        plt.subplot(nrows, ncols, crop_subplot_idx)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
            original_name = os.path.basename(original_path) if original_path else "Unknown"
            name = Path(fname).stem
            name= name[-15:]
            title = (f'#{i + 1}\n{name}\nDist: {dist:.3f}\n'
                     # f'Original: {original_name}'
                     )
            plt.title(title, fontsize=10, fontweight='bold', pad=10)
        else:
            logging.error(f'File {img_path} not found')
            plt.text(0.5, 0.5, f'Image\nNon Trouvée\n{fname}',
                     ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        plt.axis('off')

        orig_subplot_idx = crop_subplot_idx + ncols
        plt.subplot(nrows, ncols, orig_subplot_idx)
        if original_path and os.path.exists(original_path):
            orig_img = Image.open(original_path)
            plt.imshow(orig_img)
            plt.title("Image d'origine", fontsize=10, fontweight='bold', pad=10)
        else:
            plt.text(0.5, 0.5, "Image\nd'origine\nNon Trouvée",
                     ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        plt.axis('off')

    plt.suptitle(
        f'',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    logging.info(f'Saved result figure to {save_path}')
    plt.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='YOLO + ViT ear recognition pipeline')
    parser.add_argument('--path', type=str, help='Path to the image or folder')
    parser.add_argument('--reset', action='store_true', help='Reset FAISS index and NPY files')
    parser.add_argument('--add', action='store_true', help='Add image to the index after querying')
    parser.add_argument('--extract', action='store_true', help='Add cropped image in the archive directory : db/images/')
    args = parser.parse_args()
    if not args.reset and not args.path:
        parser.error("--path is required unless --reset is used")
    pipeline: EarRecognizerSystem = EarRecognizerSystem()
    if args.reset:
        pipeline.reset_index()
    else:
        if not os.path.isfile(args.path):
            files = lib.get_all_file_paths(args.path)
            files.sort()
        else:
            files = [args.path]
        for filepath in files:
            if os.path.isfile(filepath):
                pipeline.run(filepath, add=args.add, extract=args.extract)