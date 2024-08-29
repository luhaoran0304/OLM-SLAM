# Use this script to pre-train the depth / pose estimation networks

from config.config_parser import ConfigParser
from depth_pose_prediction import DepthPosePrediction

# ============================================================
config = ConfigParser('./config/config_pretrain.yaml')
print(config)

# ============================================================
predictor = DepthPosePrediction(config.dataset, config.depth_pose)
predictor.load_model(load_optimizer=False)
predictor.train(validate=False, depth_error=False, use_wandb=config.depth_pose.use_wandb)
