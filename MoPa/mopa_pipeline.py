"""
MoPa Pipeline - Motion encoder with PointNet + ViT
Separates motion encoding from text encoding and loss (handled in training code)
"""

import numpy as np
import timm
import torch
from torch import nn
from omegaconf import DictConfig
from MoPa.pointnet import PointnetTransformer


class MotionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        patch_size: int = 16,
        num_frames: int = 224,
        feature_dim: int = 512,
        feature_mode: str = "avg",  # ('', 'avg', 'avgmax', 'max', 'token', 'map')
    ) -> None:
        super().__init__()
        
        if feature_mode == 'all':
            feature_mode = ''
        # Image size: [height=num_frames, width=feature_dim (multiple of patch_size)]
        # feature_dim should be patch_size * some_value to leverage pretrained weights
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=feature_mode,
            img_size=(num_frames, feature_dim),
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        return self.model(x)


class PipelineMoPa(nn.Module):
    """
    MoPa Motion Encoder Pipeline
    Processes point cloud sequences through PointNet + ViT encoder
    """
    def __init__(
        self,
        motion_encoder_alias: str = "vit_base_patch16_224_in21k",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        patch_size: int = 16,
        num_frames: int = 16,
        num_groups: int = 64,
        point_encoder_config: dict = None,
        feature_mode: str = "all",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        
        self.device = device
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.feature_mode = feature_mode.lower()
        # Initialize PointNet + Transformer encoder for point cloud processing
        if point_encoder_config is None:
            point_encoder_config = {
                "dvae_config": {
                    "encoder_dim": 256,
                    "group_size": 32,
                    "num_group": num_groups,
                    # "encoder_out": 512,
                },
                "transformer_config": {
                    "embed_dim": 768,
                    "depth": 4,
                    "num_heads": 12,
                    "mlp_ratio": 4.0,
                    "qkv_bias": False,
                    "qk_scale": None,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "drop_path_rate": 0.1,
                },
            }
        
        dvae_config = DictConfig(point_encoder_config["dvae_config"])
        transformer_config = DictConfig(point_encoder_config["transformer_config"])
        
        self.point_encoder = PointnetTransformer(
            dvae_config=dvae_config,
            transformer_config=transformer_config,
        )
        
        # Output dimension from PointnetTransformer
        point_output_dim = dvae_config.get("encoder_out", 512)
        # Ensure feature_dim is a multiple of patch_size for pretrained weights
        feature_dim = ((point_output_dim + patch_size - 1) // patch_size) * patch_size
        
        # Store feature dimension for later use
        self.feature_dim = feature_dim
        
        # If point encoder output doesn't match required dimension, add projection
        # if point_output_dim != feature_dim:
        #     self.point_to_image = nn.Linear(point_output_dim, feature_dim)
        # else:
        #     self.point_to_image = nn.Identity()

        motion_encoder = MotionEncoder(
            model_name=motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            patch_size=patch_size,
            num_frames=num_frames,
            feature_dim=feature_dim,
            feature_mode=self.feature_mode,
        )

        self.motion_encoder = motion_encoder
        
        # Motion projection head (for contrastive learning)
        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

    def encode_motion(self, motion):
        """
        Encode motion point cloud sequences
        
        Args:
            motion: [batch, frames, points, features] - typically features=9, we use xyz only
            
        Returns:
            motion_embeddings: [batch, projection_dims] - normalized embeddings
        """
        # Dataset gives 9 features per point, we only need xyz
        motion = motion[..., :3]
        
        # motion shape: [batch, frames, points, 3]
        batch_size, frames, points, _ = motion.shape
        
        # Flatten batch and frames for efficient processing: [batch*frames, points, 3]
        motion_flat = motion.view(batch_size * frames, points, 3)
        
        # Process all frames at once through PointNet + Transformer
        # Returns: [batch*frames, 3, feature_dim] - already stacked cls, mean, max
        point_feats = self.point_encoder(motion_flat)
        
        # Project to ensure feature_dim alignment if needed
        # point_feats shape: [batch*frames, 3, feature_dim]
        batch_frames, num_channels, feat_dim = point_feats.shape
        point_feats = point_feats.view(batch_frames * num_channels, feat_dim)
        # point_feats = self.point_to_image(point_feats)
        point_feats = point_feats.view(batch_frames, num_channels, -1)  # [batch*frames, 3, feature_dim]
        
        # Reshape to separate batch and frames: [batch, frames, 3, feature_dim]
        point_feats = point_feats.view(batch_size, frames, num_channels, -1)
        
        # Permute to match image format: [batch, 3, frames, feature_dim]
        motion_image = point_feats.permute(0, 2, 1, 3)  # [batch, 3, frames, feature_dim]
        
        # Process through motion encoder
        motion_features = self.motion_encoder(motion_image)
        if self.feature_mode == "all":
            return motion_features
        motion_embeddings = self.motion_projection(motion_features)
        
        return motion_embeddings

    def forward(self, batch):
        """
        Forward pass through motion encoder
        
        Args:
            batch: [B, T, N, features] point cloud sequences
            
        Returns:
            motion_embeddings: [B, projection_dims] normalized motion embeddings
        """
        motion_embeddings = self.encode_motion(batch)
        
        # Normalize features for contrastive learning
        motion_embeddings = motion_embeddings / motion_embeddings.norm(dim=-1, keepdim=True)
        
        return motion_embeddings
    
    def get_trainable_parameters(self):
        """Return list of trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)
