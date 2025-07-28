import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import faiss
import torch
from PIL import Image
from pathlib import Path

from pipeline import EarRecognizerSystem, show_results


class TestEarRecognizerSystem:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    @pytest.fixture
    def ear_recognizer(self):
        with patch('pipeline.YOLO'), \
                patch('pipeline.ViTEmbedder'), \
                patch('pipeline.ViTImageProcessor.from_pretrained'), \
                patch('pipeline.faiss.read_index'), \
                patch('pipeline.safe_load_npy'), \
                patch('os.path.exists', return_value=True):
            return EarRecognizerSystem()

    @pytest.fixture
    def mock_pil_image(self):
        image = MagicMock(spec=Image.Image)
        image.size = (224, 224)
        return image

    @pytest.fixture
    def mock_yolo_results(self):
        results = MagicMock()
        results.boxes = MagicMock()
        results.boxes.xyxy = MagicMock()
        results.boxes.cls = MagicMock()
        results.boxes.conf = MagicMock()
        results.boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 1])
        results.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        results.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        return results

    @patch('pipeline.YOLO')
    @patch('pipeline.ViTEmbedder')
    @patch('pipeline.ViTImageProcessor.from_pretrained')
    @patch('pipeline.faiss.read_index')
    @patch('pipeline.safe_load_npy')
    @patch('os.path.exists')
    def test_init_successful(self, mock_exists, mock_safe_load_npy, mock_faiss_read,
                             mock_vit_processor, mock_vit_embedder, mock_yolo):
        mock_exists.return_value = True
        mock_safe_load_npy.side_effect = [
            np.zeros((10, 768)),
            np.zeros((10, 768)),
            np.array(['file1.jpg', 'file2.jpg']),
            np.array(['file3.jpg', 'file4.jpg']),
            np.array(['path1.jpg', 'path2.jpg']),
            np.array(['path3.jpg', 'path4.jpg'])
        ]

        recognizer = EarRecognizerSystem()

        assert recognizer.embedding_dim == 768
        assert len(recognizer.left_filenames) == 2
        assert len(recognizer.right_filenames) == 2

    def test_load_yolo_model(self, ear_recognizer, mocker):
        mock_yolo = mocker.patch('pipeline.YOLO')
        mock_log = mocker.patch('pipeline.logging.info')

        result = ear_recognizer._load_yolo_model()

        mock_yolo.assert_called_once_with(ear_recognizer.yolo_model_path)
        mock_log.assert_called_once()


    def test_load_vit_model_failure(self, ear_recognizer, mocker):
        mock_vit_embedder = mocker.patch('pipeline.ViTEmbedder')
        mock_torch_load = mocker.patch('pipeline.torch.load', side_effect=Exception("Load error"))
        mock_log_error = mocker.patch('pipeline.logging.error')

        with pytest.raises(Exception):
            ear_recognizer._load_vit_model()

        mock_log_error.assert_called_once()

    def test_safe_load_faiss_index_exists(self, ear_recognizer, mocker):
        mock_exists = mocker.patch('os.path.exists', return_value=True)
        mock_faiss_read = mocker.patch('pipeline.faiss.read_index')

        result = ear_recognizer._safe_load_faiss_index(True)

        mock_faiss_read.assert_called_once_with(ear_recognizer.faiss_index_left_path)

    def test_safe_load_faiss_index_not_exists(self, ear_recognizer, mocker):
        mock_exists = mocker.patch('os.path.exists', return_value=False)
        mock_log = mocker.patch('pipeline.logging.warning')
        mock_faiss_index = mocker.patch('pipeline.faiss.IndexFlatL2')

        result = ear_recognizer._safe_load_faiss_index(False)

        mock_faiss_index.assert_called_once_with(ear_recognizer.embedding_dim)
        mock_log.assert_called_once()

    def test_reset_index(self, ear_recognizer, mocker):
        mock_faiss_write = mocker.patch('pipeline.faiss.write_index')
        mock_faiss_index = mocker.patch('pipeline.faiss.IndexFlatL2')
        mock_np_save = mocker.patch('numpy.save')
        mock_log_warning = mocker.patch('pipeline.logging.warning')
        mock_log_info = mocker.patch('pipeline.logging.info')

        ear_recognizer.reset_index()

        assert mock_faiss_write.call_count == 2
        assert mock_np_save.call_count == 6
        mock_log_warning.assert_called_once()
        mock_log_info.assert_called_once()

    def test_create_crops_with_save(self, ear_recognizer, mocker):
        mock_makedirs = mocker.patch('os.makedirs')
        mock_exists = mocker.patch('os.path.exists', return_value=False)
        mock_log = mocker.patch('pipeline.logging.info')

        class_index = [0]
        confs = [0.9]
        boxes = [[10, 10, 50, 50]]
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil = MagicMock()
            mock_from_array.return_value = mock_pil

            crops = ear_recognizer.create_crops(
                class_index, confs, boxes, img_array, True, 'output_dir', 'test.jpg'
            )

        assert len(crops) == 1
        mock_pil.save.assert_called_once()
        mock_makedirs.assert_called_once()

    def test_create_crops_without_save(self, ear_recognizer, mocker):
        mock_makedirs = mocker.patch('os.makedirs')
        mock_exists = mocker.patch('os.path.exists', return_value=False)

        class_index = [0]
        confs = [0.9]
        boxes = [[10, 10, 50, 50]]
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil = MagicMock()
            mock_from_array.return_value = mock_pil

            crops = ear_recognizer.create_crops(
                class_index, confs, boxes, img_array, False, 'output_dir', 'test.jpg'
            )

        assert len(crops) == 1
        mock_pil.save.assert_not_called()

    def test_extract_ear_from_image_no_boxes(self, ear_recognizer, mocker):
        mock_results = MagicMock()
        mock_results.boxes = None
        mocker.patch.object(ear_recognizer.yolo_model, '__call__', return_value=[mock_results])
        mock_log = mocker.patch('pipeline.logging.error')

        left_crops, right_crops = ear_recognizer.extract_ear_from_image('test.jpg')

        assert left_crops == []
        assert right_crops == []
        mock_log.assert_called_once()

    def test_extract_ear_from_image_empty_boxes(self, ear_recognizer, mocker):
        mock_results = MagicMock()
        mock_results.boxes = MagicMock()
        mock_results.boxes.xyxy = []
        mocker.patch.object(ear_recognizer.yolo_model, '__call__', return_value=[mock_results])
        mock_log = mocker.patch('pipeline.logging.error')

        left_crops, right_crops = ear_recognizer.extract_ear_from_image('test.jpg')

        assert left_crops == []
        assert right_crops == []
        mock_log.assert_called_once()

    def test_compute_embedding(self, ear_recognizer, mock_pil_image, mocker):
        mock_processor = mocker.patch.object(ear_recognizer, 'processor')
        mock_vit_model = mocker.patch.object(ear_recognizer, 'vit_model')

        mock_processor.return_value = {'pixel_values': torch.zeros(1, 3, 224, 224)}
        mock_vit_model.return_value.cpu.return_value.numpy.return_value = np.zeros((1, 768))

        embedding = ear_recognizer.compute_embedding(mock_pil_image)

        assert isinstance(embedding, np.ndarray)
        mock_processor.assert_called_once_with(mock_pil_image, return_tensors='pt')

    def test_search_similar_successful(self, ear_recognizer, mock_pil_image, mocker):
        ear_recognizer.faiss_left = MagicMock()
        ear_recognizer.faiss_left.ntotal = 5
        ear_recognizer.faiss_left.search.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))
        ear_recognizer.left_filenames = ['file1.jpg', 'file2.jpg', 'file3.jpg']
        ear_recognizer.left_original_paths = ['path1.jpg', 'path2.jpg', 'path3.jpg']

        mocker.patch.object(ear_recognizer, 'compute_embedding', return_value=np.zeros((1, 768)))
        mock_log = mocker.patch('pipeline.logging.info')

        results = ear_recognizer.search_similar(mock_pil_image, top_k=3, left=True)

        assert len(results) == 3
        assert results[0] == ('file1.jpg', 0.1, 'path1.jpg')
        assert mock_log.call_count == 3

    def test_search_similar_empty_index(self, ear_recognizer, mock_pil_image, mocker):
        ear_recognizer.faiss_left = MagicMock()
        ear_recognizer.faiss_left.ntotal = 0
        mock_log = mocker.patch('pipeline.logging.error')

        results = ear_recognizer.search_similar(mock_pil_image, top_k=3, left=True)

        assert results == []
        mock_log.assert_called_once_with('No faiss index found')

    def test_search_similar_index_out_of_bounds(self, ear_recognizer, mock_pil_image, mocker):
        ear_recognizer.faiss_left = MagicMock()
        ear_recognizer.faiss_left.ntotal = 5
        ear_recognizer.faiss_left.search.return_value = (np.array([[0.1]]), np.array([[10]]))
        ear_recognizer.left_filenames = ['file1.jpg', 'file2.jpg']

        mocker.patch.object(ear_recognizer, 'compute_embedding', return_value=np.zeros((1, 768)))
        mock_log = mocker.patch('pipeline.logging.error')

        results = ear_recognizer.search_similar(mock_pil_image, top_k=1, left=True)

        assert results == []
        mock_log.assert_called_once()

    def test_add_to_faiss_left(self, ear_recognizer, mock_pil_image, mocker):
        ear_recognizer.faiss_left = MagicMock()
        ear_recognizer.embeddings_left = np.zeros((0, 768))
        ear_recognizer.left_filenames = []
        ear_recognizer.left_original_paths = []

        mocker.patch.object(ear_recognizer, 'compute_embedding', return_value=np.ones((1, 768)))
        mock_np_save = mocker.patch('numpy.save')
        mock_faiss_write = mocker.patch('pipeline.faiss.write_index')
        mock_log = mocker.patch('pipeline.logging.info')

        ear_recognizer.add_to_faiss(mock_pil_image, 'test.jpg', 'original.jpg', True)

        ear_recognizer.faiss_left.add.assert_called_once()
        assert mock_np_save.call_count == 3
        mock_faiss_write.assert_called_once()
        mock_log.assert_called_once()

    def test_add_to_faiss_right(self, ear_recognizer, mock_pil_image, mocker):
        ear_recognizer.faiss_right = MagicMock()
        ear_recognizer.embeddings_right = np.zeros((0, 768))
        ear_recognizer.right_filenames = []
        ear_recognizer.right_original_paths = []

        mocker.patch.object(ear_recognizer, 'compute_embedding', return_value=np.ones((1, 768)))
        mock_np_save = mocker.patch('numpy.save')
        mock_faiss_write = mocker.patch('pipeline.faiss.write_index')

        ear_recognizer.add_to_faiss(mock_pil_image, 'test.jpg', 'original.jpg', False)

        ear_recognizer.faiss_right.add.assert_called_once()

    @patch('pipeline.show_results')
    def test_run_successful_extraction_and_search(self, mock_show_results, ear_recognizer, mocker):
        mocker.patch.object(ear_recognizer, 'extract_ear_from_image', return_value=([('crop1', 'path1')], []))
        mocker.patch.object(ear_recognizer, 'search_similar', return_value=[('file1', 0.1, 'orig_path1')])

        ear_recognizer.run('test_image.jpg')

        ear_recognizer.extract_ear_from_image.assert_called_once_with('test_image.jpg', save_cropped=False)
        ear_recognizer.search_similar.assert_called_once()
        mock_show_results.assert_called_once()

    @patch('pipeline.show_results')
    def test_run_successful_add_to_faiss(self, mock_show_results, ear_recognizer, mocker):
        mocker.patch.object(ear_recognizer, 'extract_ear_from_image', return_value=([('crop1', 'path1')], []))
        mocker.patch.object(ear_recognizer, 'search_similar', return_value=[('file1', 0.1, 'orig_path1')])
        mocker.patch.object(ear_recognizer, 'add_to_faiss')

        ear_recognizer.run('test_image.jpg', add=True)

        ear_recognizer.add_to_faiss.assert_called_once()

    @patch('pipeline.show_results')
    def test_run_successful_save_cropped_images(self, mock_show_results, ear_recognizer, mocker):
        mocker.patch.object(ear_recognizer, 'extract_ear_from_image', return_value=([('crop1', 'path1')], []))
        mocker.patch.object(ear_recognizer, 'search_similar', return_value=[('file1', 0.1, 'orig_path1')])

        ear_recognizer.run('test_image.jpg', extract=True)

        ear_recognizer.extract_ear_from_image.assert_called_once_with('test_image.jpg', save_cropped=True)

    def test_run_no_ears_detected(self, ear_recognizer, mocker):
        mocker.patch.object(ear_recognizer, 'extract_ear_from_image', return_value=([], []))
        mock_log = mocker.patch('pipeline.logging.critical')

        ear_recognizer.run('test_image.jpg')

        mock_log.assert_called_once_with('No ears found in test_image.jpg')

    @patch('pipeline.show_results')
    def test_run_both_ears_detected(self, mock_show_results, ear_recognizer, mocker):
        mocker.patch.object(ear_recognizer, 'extract_ear_from_image',
                            return_value=([('left_crop', 'left_path')], [('right_crop', 'right_path')]))
        mocker.patch.object(ear_recognizer, 'search_similar', return_value=[('file1', 0.1, 'orig_path1')])

        ear_recognizer.run('test_image.jpg')

        assert ear_recognizer.search_similar.call_count == 2
        assert mock_show_results.call_count == 2


class TestShowResults:

    @pytest.fixture
    def mock_pil_image(self):
        return MagicMock(spec=Image.Image)

    def test_show_results_successful(self, mock_pil_image, mocker):
        mock_makedirs = mocker.patch('os.makedirs')
        mock_listdir = mocker.patch('os.listdir', return_value=['1.png', '2.png'])
        mock_exists = mocker.patch('os.path.exists', return_value=True)
        mock_image_open = mocker.patch('PIL.Image.open', return_value=mock_pil_image)

        with patch('matplotlib.pyplot.figure'), \
                patch('matplotlib.pyplot.subplot'), \
                patch('matplotlib.pyplot.imshow'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.axis'), \
                patch('matplotlib.pyplot.suptitle'), \
                patch('matplotlib.pyplot.savefig') as mock_savefig, \
                patch('matplotlib.pyplot.close'), \
                patch('matplotlib.pyplot.subplots_adjust'):
            mock_log = mocker.patch('pipeline.logging.info')

            show_results(
                mock_pil_image,
                [('file1.jpg', 0.1, 'original1.jpg'), ('file2.jpg', 0.2, 'original2.jpg')],
                'test_image',
                'db_folder',
                'save_dir'
            )

            mock_savefig.assert_called_once()
            mock_log.assert_called_once()

    def test_show_results_missing_image(self, mock_pil_image, mocker):
        mock_makedirs = mocker.patch('os.makedirs')
        mock_listdir = mocker.patch('os.listdir', return_value=[])
        mock_exists = mocker.patch('os.path.exists', return_value=False)
        mock_log_error = mocker.patch('pipeline.logging.error')

        with patch('matplotlib.pyplot.figure'), \
                patch('matplotlib.pyplot.subplot'), \
                patch('matplotlib.pyplot.imshow'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.axis'), \
                patch('matplotlib.pyplot.text'), \
                patch('matplotlib.pyplot.suptitle'), \
                patch('matplotlib.pyplot.savefig'), \
                patch('matplotlib.pyplot.close'), \
                patch('matplotlib.pyplot.subplots_adjust'):
            show_results(
                mock_pil_image,
                [('missing_file.jpg', 0.1, 'original1.jpg')],
                'test_image',
                'db_folder',
                'save_dir'
            )

            mock_log_error.assert_called_once()

    def test_show_results_with_tuple_length_2(self, mock_pil_image, mocker):
        mock_makedirs = mocker.patch('os.makedirs')
        mock_listdir = mocker.patch('os.listdir', return_value=[])
        mock_exists = mocker.patch('os.path.exists', return_value=True)
        mock_image_open = mocker.patch('PIL.Image.open', return_value=mock_pil_image)

        with patch('matplotlib.pyplot.figure'), \
                patch('matplotlib.pyplot.subplot'), \
                patch('matplotlib.pyplot.imshow'), \
                patch('matplotlib.pyplot.title'), \
                patch('matplotlib.pyplot.axis'), \
                patch('matplotlib.pyplot.suptitle'), \
                patch('matplotlib.pyplot.savefig'), \
                patch('matplotlib.pyplot.close'), \
                patch('matplotlib.pyplot.subplots_adjust'):
            show_results(
                mock_pil_image,
                [('file1.jpg', 0.1)],
                'test_image',
                'db_folder',
                'save_dir'
            )