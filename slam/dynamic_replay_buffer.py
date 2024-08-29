import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import faiss
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from slam.similar import histogram, pHash
from datasets.utils import get_random_color_jitter
from slam.static_replay_buffer import StaticReplayBuffer
from collections import deque


class DynamicReplayBuffer(TorchDataset):
    def __init__(
            self,
            storage_dir: Path,
            dataset_type: str,
            state_path: Optional[Path] = None,
            height: int = 0,
            width: int = 0,
            scales: List[int] = None,
            frames: List[int] = None,
            num_workers: int = 1,
            do_augmentation: bool = False,
            batch_size: int = 1,
            maximize_diversity: bool = True,
            dynamic_buffer_size: int = 6,
            similarity_threshold: float = 1,
            similarity_sampling: bool = True,
    ):
        self.storage_dir = storage_dir

        self.dataset_type = dataset_type.lower()
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation
        self.batch_size = batch_size

        # Restrict size of the replay buffer
        self.NUMBER_SAMPLES_PER_ENVIRONMENT = 100
        self.valid_indices = {}

        self.buffer_filenames = {}
        self.online_filenames = []

        # Precompute the resize functions for each scale relative to the previous scale
        # If scales is None, the size of the raw data will be used
        self.scales = scales
        self.frames = frames
        self.resize = {}
        if self.scales is not None:
            for s in self.scales:
                exp_scale = 2 ** s
                self.resize[s] = transforms.Resize(
                    (height // exp_scale, width // exp_scale),
                    interpolation=transforms.InterpolationMode.LANCZOS)

        # Ensure repeatability of experiments
        self.target_sampler = np.random.default_rng(seed=42)

        # Dissimilarity-based buffer
        self.similarity_sampling = similarity_sampling
        self.maximize_diversity = maximize_diversity
        self.dynamic_buffer_size = dynamic_buffer_size
        self.similarity_threshold = similarity_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.faiss_index = None
        self.faiss_index_offset = 0
        self.distance_matrix = None
        self.distance_matrix_indices = None
        self.faiss_phash = None
        self.phash = []
        self.fsd = FixedSizeDeque(max_size = dynamic_buffer_size)

    def dynamic_add(self, sample: Dict[str, Any], sample_filenames: Dict[str, Any],
                    staticreplaybuffer: StaticReplayBuffer, histograms: Optional[np.ndarray] = None,
                    verbose: bool = False):
        index = sample['index'].item()
        assert index == sample_filenames['index']

        index += self.faiss_index_offset

        if self.faiss_index is None:
            if histograms is None:
                num_histograms = 256 * 3
            else:
                num_histograms = histograms.shape[1]
            self.faiss_index = faiss.IndexIDMap(
                faiss.index_factory(num_histograms, 'Flat', faiss.METRIC_INNER_PRODUCT))
        if histograms is None:
            histograms = histogram(sample['rgb', 0, 0])
        faiss.normalize_L2(histograms)  # The inner product becomes cosine similarity

        add_sample = False
        remove_sample = None
        if self.maximize_diversity:
            # Only add if sufficiently dissimilar to the existing samples
            if self.faiss_index.ntotal == 0:
                similarity = 0
            else:
                similarity = self.faiss_index.search(histograms, 1)[0][0][0]

            if similarity < self.similarity_threshold:
                old_index = self.fsd.append(index)
                self.faiss_index.add_with_ids(histograms, np.array([index]))
                add_sample = True
                if verbose:
                    print(f'Added sample {index} to the dynamic replay buffer | similarity {similarity}')

                if old_index is not None:
                    self.faiss_index.remove_ids(np.array([old_index]))
                    remove_sample = old_index
                    # if verbose:
                    #     print(f'Removed sample {remove_index} from the dynamic replay buffer')

        else:
            self.faiss_index.add_with_ids(histograms, np.array([index]))
            add_sample = True
            if self.faiss_index.ntotal > self.dynamic_buffer_size:
                remove_index = self.target_sampler.choice(self.faiss_index.ntotal, 1)[0]
                remove_sample = faiss.vector_to_array(self.faiss_index.id_map)[remove_index]
                self.faiss_index.remove_ids(np.array([remove_sample]))
                # if verbose:
                #     print(f'Removed sample {remove_sample} from the dynamic buffer')

        if add_sample:
            filename = self.storage_dir / f'{self.dataset_type}_{index:>05}.pkl'
            data = {
                key: value
                for key, value in sample.items() if 'index' in key or 'camera_matrix' in key
                                                    or 'inv_camera_matrix' in key
                                                    or 'relative_distance' in key
            }
            data['rgb', -1] = sample_filenames['images'][0]
            data['rgb', 0] = sample_filenames['images'][1]
            data['rgb', 1] = sample_filenames['images'][2]
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            self.online_filenames.append(filename)

        if remove_sample is not None:
            for filename in self.online_filenames:
                if f'_{remove_sample:>05}.pkl' in filename.name:
                    remove_date = self._get(filename)
                    remove_filenames = self._get_filennames(remove_sample, filename)
                    staticreplaybuffer.static_add(remove_date, remove_filenames, verbose=verbose)
                    os.remove(filename)
                    self.online_filenames.remove(filename)
                    break

    def dynamic_get(self, sample: Dict[str, Any], histograms: Optional[np.ndarray] = None,) -> Dict[
        str, Any]:
        return_data = {}

        # Sample from target buffer
        if self.online_filenames and self.batch_size > 0:
            index = sample['index'].item() + self.faiss_index_offset
            filename = self.storage_dir / f'{self.dataset_type}_{index:>05}.pkl'
            # The current sample is the only one that is in the buffer
            if len(self.online_filenames) == 1 and filename in self.online_filenames:
                replace = True
                num_samples = 1
                sampling_prob = None
            else:
                # Do not sample the current sample
                if filename in self.online_filenames:
                    num_samples = len(self.online_filenames) - 1  # -1 for the current sample
                else:
                    num_samples = len(self.online_filenames)
                replace = self.batch_size > num_samples

                if self.similarity_sampling:
                    assert self.faiss_index.ntotal > 0
                    if histograms is None:
                        histograms = histogram(sample['rgb', 0, 0])
                    faiss.normalize_L2(
                        histograms)  # The inner product becomes cosine similarity
                    similarity, indices = self.faiss_index.search(histograms,
                                                                  self.faiss_index.ntotal)
                    if index in indices:
                        similarity = np.delete(similarity, np.argwhere(indices == index))
                    else:
                        similarity = similarity[0]
                    dissimilarity = 1 - similarity
                    # sampling_prob = dissimilarity / dissimilarity.sum()
                    sampling_prob = similarity / similarity.sum()
                else:
                    sampling_prob = None

            indices = self.target_sampler.choice(num_samples, 1, replace,
                                                 sampling_prob)
            filenames = [self.online_filenames[index] for index in indices]
            return_data = self._get(filenames[0])
            for filename in filenames[1:]:
                data = self._get(filename)
                for key in return_data:
                    return_data[key] = torch.cat([return_data[key], data[key]])

        return return_data

    def save_state(self):
        filename = self.storage_dir / 'buffer_state.pkl'
        data = {'filenames': self.online_filenames, 'faiss_index': self.faiss_index}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved reply buffer state to: {filename}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

    def load_state(self, state_path: Path):
        with open(state_path, 'rb') as f:
            data = pickle.load(f)
            # self.buffer_filenames = data['filenames']
            self.faiss_index = data['faiss_index']
            self.faiss_index_offset = faiss.vector_to_array(self.faiss_index.id_map).max()
            self.online_filenames = [state_path.parent / file.name for file in data['filenames']]
        print(f'Load replay buffer state from: {state_path}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        raise NotImplementedError

    def __len__(self):
        return 1000000  # Fixed number as the sampling is handled in the get() function

    def _get(self, filename, include_batch=True):
        if self.do_augmentation:
            color_augmentation = get_random_color_jitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2),
                                                         (-.1, .1))

        with open(filename, 'rb') as f:
            data = pickle.load(f)
        for frame in self.frames:
            rgb = Image.open(data['rgb', frame]).convert('RGB')
            rgb = self.resize[0](rgb)
            data['rgb', frame, 0] = rgb
            for scale in self.scales:
                if scale == 0:
                    continue
                data['rgb', frame, scale] = self.resize[scale](data['rgb', frame, scale - 1])
            for scale in self.scales:
                data['rgb', frame, scale] = transforms.ToTensor()(data['rgb', frame, scale])
                if include_batch:
                    data['rgb', frame, scale] = data['rgb', frame, scale].unsqueeze(0)
                if self.do_augmentation:
                    data['rgb_aug', frame, scale] = color_augmentation(data['rgb', frame, scale])
                else:
                    data['rgb_aug', frame, scale] = data['rgb', frame, scale]
            del data['rgb', frame]  # Removes the filename string
        if not include_batch:
            for key in data:
                if not ('rgb' in key or 'rgb_aug' in key):
                    data[key] = data[key].squeeze(0)
        return data

    def _get_filennames(self, index, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        img_filenames = []
        mask_filenames = []
        img_filenames.append(data['rgb', -1])
        img_filenames.append(data['rgb', 0])
        img_filenames.append(data['rgb', 1])
        filenames = {
            'index': index,
            'images': img_filenames,
            'masks': mask_filenames,
        }
        return filenames

    def _reset_storage_dir(self):
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def release_dynamic_buffer(self, staticreplaybuffer: StaticReplayBuffer):
        for filename in self.online_filenames:
            matches = re.findall(r'\d', filename.name)
            i = 0
            for index, char in enumerate(matches):
                if int(char) != 0:
                    i = index
                    break
            ids_str = ''.join(matches[i:])
            ids = int(ids_str)
            remove_date = self._get(filename)
            remove_filenames = self._get_filennames(ids, filename)
            staticreplaybuffer.static_add(remove_date, remove_filenames, verbose=False)
            os.remove(filename)
        self.faiss_index.reset()
        print("Number of vectors in the index:", self.faiss_index.ntotal)

    def dynamic_buffer_stocks(self) -> bool:
        if self.faiss_index.ntotal == self.dynamic_buffer_size:
            return True
        else:
            return False


class FixedSizeDeque:
    def __init__(self, max_size):
        self.deque = deque(maxlen=max_size)

    def append(self, item):
        if len(self.deque) == self.deque.maxlen:
            popped_item = self.deque.popleft()
            self.deque.append(item)
            return popped_item
        else:
            self.deque.append(item)
            return None

    def get_oldest(self):
        if self.deque:
            return self.deque[0]
        else:
            return None

    def remove_oldest(self):
        if self.deque:
            return self.deque.popleft()
        else:
            return None

    def __len__(self):
        return len(self.deque)