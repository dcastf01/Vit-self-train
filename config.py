import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass

import ml_collections
# from models.vt.configs import ConfigsVit

@dataclass
class CONFIG:

    ROOT_WORKSPACE: str=""
        
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    BATCH_SIZE:int = 128
    NUM_CLASSES:int=4455
    LEARNING_RATE:float = 1e-5
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    IMG_SIZE:int=224
    NUM_EPOCHS :int= 800
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"models/checkpoints")
    

    

@dataclass
class ModelCfg:
    pass
    # backbone_name:Enum=ConfigsVit.ViT_B_16