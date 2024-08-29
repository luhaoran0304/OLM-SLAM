import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Slam:
    config_file: Path
    dataset_sequence: int
    adaptation: bool
    adaptation_epochs: int
    min_distance: float
    start_frame: int
    logging: bool
    do_loop_closures: bool
    keyframe_frequency: int
    lc_distance_poses: int

@dataclasses.dataclass
class ReplayBuffer:
    config_file: Path
    maximize_diversity: bool
    dynamic_max_buffer_size: int
    static_max_buffer_size: int
    dynamic_similarity_threshold: float
    static_similarity_threshold: float
    similarity_sampling: bool
    dynamic_load_path: Path
    static_load_path: Path
