from typing import List, Tuple
import faiss
import numpy as np
import torch
from torch import Tensor

from loop_closure_detection.config import LoopClosureDetection as Config
import cv2

class LoopClosureDetection:
    def __init__(
        self,
        config: Config,
    ):
        # Initialize parameters ===========================
        self.threshold = config.detection_threshold
        self.id_threshold = config.id_threshold
        self.num_matches = config.num_matches

        # Fixed parameters ================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # =================================================

        # Feature cache ===================================
        # Cosine similarity
        self.faiss_index = faiss.index_factory(64, 'Flat',
                                               faiss.METRIC_INNER_PRODUCT)
        self.image_id_to_index = {}
        self.index_to_image_id = {}
        # =================================================

    def add(self, image_id: int, image: Tensor) -> None:
        # Add batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)
        features = self.pHash(image)
        faiss.normalize_L2(features)  # Then the inner product becomes cosine similarity
        self.faiss_index.add(features)
        self.image_id_to_index[image_id] = self.faiss_index.ntotal - 1
        self.index_to_image_id[self.faiss_index.ntotal - 1] = image_id
        # print(f'Is Faiss index trained: {self.faiss_index.is_trained}')

    def search(self, image_id: int) -> Tuple[List[int], List[float]]:
        index_id = self.image_id_to_index[image_id]
        features = np.expand_dims(self.faiss_index.reconstruct(index_id), 0)
        distances, indices = self.faiss_index.search(features, 100)
        distances = distances.squeeze()
        indices = indices.squeeze()
        # Remove placeholder entries without a match
        distances = distances[indices != -1]
        indices = indices[indices != -1]
        # Remove self detection
        distances = distances[indices != index_id]
        indices = indices[indices != index_id]
        # Filter by the threshold
        indices = indices[distances > self.threshold]
        distances = distances[distances > self.threshold]
        # Do not return neighbors (trivial matches)
        distances = distances[np.abs(indices - index_id) > self.id_threshold]
        indices = indices[np.abs(indices - index_id) > self.id_threshold]
        # Return best N matches
        distances = distances[:self.num_matches]
        indices = indices[:self.num_matches]
        # Convert back to image IDs
        image_ids = sorted([self.index_to_image_id[index_id] for index_id in indices])
        return image_ids, distances


    def pHash(self, image):
        image_np = image.detach().cpu().numpy()
        if image_np.ndim == 4:
            image_np = image_np[0]
        image_np = image_np.transpose(1, 2, 0)
        if image_np.ndim == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image_np = cv2.resize(image_np, (32, 32), interpolation=cv2.INTER_CUBIC)

        dct = cv2.dct(np.float32(image_np))
        dct_roi = dct[0:8, 0:8]
        average = np.mean(dct_roi)
        hash = []
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > average:
                    hash.append(1)
                else:
                    hash.append(0)
        hash_np = np.array(hash, dtype=np.float32).reshape(1, -1)
        return hash_np