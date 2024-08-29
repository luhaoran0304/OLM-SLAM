from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Cityscapes
from slam.dynamic_replay_buffer import DynamicReplayBuffer
from slam.static_replay_buffer import StaticReplayBuffer
from config.config_parser import ConfigParser

# ============================================================

config = ConfigParser('./config/config_adapt.yaml')
dynamic_replay_buffer_path = Path(
    __file__).parent / './log/replay_buffer/dynamic_replay_buffer'  # <-- MATCH WITH config_pretrain.yaml
dynamic_replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
static_replay_buffer_path = Path(
    __file__).parent / './log/replay_buffer/static_replay_buffer'  # <-- MATCH WITH config_pretrain.yaml
static_replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
dynamic_replay_buffer = DynamicReplayBuffer(dynamic_replay_buffer_path, 'Cityscapes',
                                            height=config.dataset.height,
                                            width=config.dataset.width,
                                            scales=config.dataset.scales,
                                            frames=config.dataset.frame_ids,
                                            dynamic_buffer_size=config.replay_buffer.dynamic_max_buffer_size,
                                            similarity_threshold=0.95)
static_replay_buffer = StaticReplayBuffer(static_replay_buffer_path, 'Cityscapes',
                                          height=config.dataset.height,
                                          width=config.dataset.width,
                                          scales=config.dataset.scales,
                                          frames=config.dataset.frame_ids,
                                          # static_buffer_size=config.replay_buffer.static_max_buffer_size,
                                          similarity_threshold=0.95)

# ============================================================

dataset = Cityscapes(
    Path('USER/data/cityscapes'),  # <-- ADJUST THIS
    'train',
    [-1, 0, 1],
    [0, 1, 2, 3],
    192,
    640,
    do_augmentation=False,
    views=('left',),
)
dataloader = DataLoader(dataset, num_workers=16, batch_size=1, shuffle=False, drop_last=True)

# ============================================================
dynamic_replay_buffer._reset_storage_dir()
static_replay_buffer._reset_storage_dir()
with tqdm(total=len(dataloader)) as pbar:
    for i, sample in enumerate(dataloader):
        dynamic_replay_buffer.dynamic_add(sample, dataset.get_item_filenames(i),
                                          static_replay_buffer)
        # print(dataset.get_item_filenames(i))
        pbar.update(1)
# dynamic_replay_buffer.save_state()
dynamic_replay_buffer.release_dynamic_buffer(static_replay_buffer)
static_replay_buffer.save_state()
