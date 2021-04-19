#https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn, einsum
import torch.nn.functional as F
import ml_collections
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class PoolLastLayer(nn.Module):
    def __init__ (self,pool):
        super().__init__()
        self.pool = pool
    def forward(self,x):
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x
    

    
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        self.to_patch_embedding=PatchEmbedding(channels,patch_size,dim,img_size=image_size)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = PoolLastLayer(pool)
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)
        # b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        x = self.transformer(x)

        x=self.pool(x)

        x = self.to_latent(x)
        return self.mlp_head(x)
    

def _get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def _get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def _get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = _get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config

def _get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = _get_b16_config()
    config.patches.size = (32, 32)
    return config

def _get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def _get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = _get_l16_config()
    config.patches.size = (32, 32)
    return config

def _get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


  
def VisionTransformerGenerator(name:str='ViT-B_16',
                               img_size=224,
                               num_classes=21843, 
                               zero_head=False,
                               vis=False
                               ):

    """Builds and returns the specified ResNet.
    Args:
        name:
            ViT version from ViT-{B_16,B_32,L_16,L32,H_14} or [R50-ViT-B_16 ,testing].
        img_size:
            size image.
        num_classes:
            Output dim of the last layer.
        zero_head:
            terminar
        vis:
            Terminar
    Returns:
        Vit as nn.Module.
    Examples:
        >>> # binary classifier with ViT-B_16
        >>> from lightly.models import VisionTransformerGenerator
        >>> vit = VisionTransformerGenerator('ViT-B_16', num_classes=2)

    """
    
    CONFIGS_VIT = {
        'ViT-B_16': _get_b16_config(),
        'ViT-B_32': _get_b32_config(),
        'ViT-L_16': _get_l16_config(),
        'ViT-L_32': _get_l32_config(),
        'ViT-H_14': _get_h14_config(),
        'R50-ViT-B_16': _get_r50_b16_config(),
        'testing': _get_testing(),
            }
    config=CONFIGS_VIT[name]
    
    if name not in CONFIGS_VIT.keys():
        raise ValueError('Illegal name: {%s}. \
        Try ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14, R50-ViT-B_16.' % (name))
    patch_size=config.patches["size"][0]
    return ViT  (
                 image_size=img_size, 
                 patch_size=patch_size, 
                 num_classes=num_classes, 
                 dim=config.hidden_size, 
                 depth=config.transformer.num_layers, 
                 heads=config.transformer.num_heads, 
                 mlp_dim=config.transformer.mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64,
                 dropout = config.transformer.attention_dropout_rate,
                 emb_dropout=config.transformer.dropout_rate
                            
                             )
    