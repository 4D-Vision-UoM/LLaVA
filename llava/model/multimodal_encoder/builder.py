import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .custom_vision_encoder import CustomVisionTower
from .mopa_encoder import MopaVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    
    # Check if using MoPa motion encoder
    if "mopa" in vision_tower.lower() or "motion" in vision_tower.lower():
        print(f"Building MoPa motion encoder: {vision_tower}")
        return MopaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # Check if using custom vision encoder
    if vision_tower.startswith("custom-vision") or "custom" in vision_tower.lower():
        print(f"Building custom vision tower: {vision_tower}")
        return CustomVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # Standard CLIP-based encoders
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
