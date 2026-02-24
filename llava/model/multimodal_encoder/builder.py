import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    if "motionpointnet" in vision_tower.lower():
        print(f"Building MotionPointNet motion encoder: {vision_tower}")
        from .motion_point_net import MotionPointNetVisionTower
        return MotionPointNetVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "psttransformer" in vision_tower.lower():
        print(f"Building PSTTransformer motion encoder: {vision_tower}")
        from .psttransformer import PSTTransformerVisionTower
        return PSTTransformerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "p4transformer" in vision_tower.lower():
        print(f"Building P4Transformer motion encoder: {vision_tower}")
        from .p4transformer import P4TransformerVisionTower
        return P4TransformerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if "mopa" in vision_tower.lower():
        from .mopa_encoder import MopaVisionTower
        print(f"Building MoPa motion encoder: {vision_tower}")
        return MopaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    
    # Standard CLIP-based encoders
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
