import torch
# from models.vt.configs import ConfigsVit
from models.vt.vit import VisionTransformer,CONFIGS_VIT
from models.simsiam.simsiam import SimSiam
from torchvision.models import resnet50

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_model(model_cfg):
   
    if model_cfg.backbone_name in ConfigsVit:
        
        backbone=VisionTransformer(CONFIGS_VIT[model_cfg.backbone_name],img_size=32, vis=False)
        backbone.output_dim=backbone.head.in_features
        backbone.head=torch.nn.Identity()
        
        model=SimSiam(backbone)
        
        return model

 