#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # TODO: Change this to correct gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict
import importlib
import json
from typing import Dict, List, Optional, Tuple, Any


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from src.datasets.humanML import HumanMLDataset, collate_fn
from src.models.text_models import create_text_encoder
from src.models.learning_objectives import TextActionAlignment
from src.utils.train_utils import (
    ConfigManager, setup_training_environment, TrainingResumer, 
    MetricsTracker, create_training_state_dict, ValidationEvaluator
)
from transformers import BertTokenizer, BertModel


import transformers
from transformers import AutoTokenizer
import numpy as np

# TODO: transfer this dataset class to HumanML dataset class python file once the implementation is finihsed
class HumanMLVQADataset(HumanMLDataset):
    """HumanML dataset for VQA"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def _discover_samples(self) -> List[Dict]:
        """
        Recursively finds all 'report.json' files in the split directory 
        and indexes them as samples.
        """
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Find all report.json files
        report_files = sorted(list(split_dir.rglob('report.json')))

        # QnA json file paths 

        qna_files = sorted(list(split_dir.rglob('*_qna.json')))
        detailed_caption_files = sorted(list(split_dir.rglob('*.txt')))
        video_files = sorted(list(split_dir.rglob('*.mp4')))
        webm_files = sorted(list(split_dir.rglob('*.webm')))

        report_files_paths_dict = {report_file.parent.name: report_file for report_file in report_files}
        qna_files_paths_dict = {qna_file.parent.name: qna_file for qna_file in qna_files}
        detailed_caption_files_paths_dict = {detailed_caption_file.parent.name: detailed_caption_file for detailed_caption_file in detailed_caption_files}
        video_files_paths_dict = {video_file.parent.name: video_file for video_file in video_files}
        webm_files_paths_dict = {webm_file.parent.name: webm_file for webm_file in webm_files}

        # Delete the lists 
        del report_files
        del qna_files
        del detailed_caption_files
        del video_files
        del webm_files

        samples = []
        # for report_path in report_files:
        for sequence_name, qna_file_path in qna_files_paths_dict.items():
            seq_dir = qna_file_path.parent
            
            # Find all PCD frames
            frame_files = sorted(list(seq_dir.glob('frame_*.pcd')))
            if not frame_files:
                continue

            # Parse frame numbers for sorting/sampling
            # Assuming format "frame_XXX.pcd"
            frames = []
            for f in frame_files:
                try:
                    num = int(f.stem.split('_')[1])
                    frames.append((num, str(f)))
                except (IndexError, ValueError):
                    pass
            
            frames.sort(key=lambda x: x[0])
            
            if not frames:
                continue

            samples.append({
                'report_path': str(report_files_paths_dict[sequence_name]),
                'qna_path': str(qna_file_path),
                'detailed_caption_path': str(detailed_caption_files_paths_dict[sequence_name]),
                'video_path': str(video_files_paths_dict[sequence_name]),
                'webm_path': str(webm_files_paths_dict[sequence_name]),

                'frames': frames, # List of (frame_num, file_path)
                'total_frames': len(frames),
                'sequence_name': seq_dir.name
            })
            
        return samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns one full action segment with windowed sampling.
        """
        sample_info = self.samples[idx]
        all_frames = sample_info['frames']
        

        # Load detailed caption from detailed_caption_path
        detailed_caption = ""
        try:
            with open(sample_info['detailed_caption_path'], 'r') as f:
                detailed_caption = f.read()
        except Exception:
            print(f"Error loading detailed caption from {sample_info['detailed_caption_path']}")
            print(e)
            pass

        # Load qna from qna_path
        qna = ""
        try:
            # Check whether the file exists
            if not os.path.exists(sample_info['qna_path']):
                raise FileNotFoundError(f"QnA file not found: {sample_info['qna_path']}")

            with open(sample_info['qna_path'], 'r') as f:
                qna_data = json.load(f)
        except Exception as e:
            print(f"Error loading qna from {sample_info['qna_path']}")
            print(e)
            pass

        # 1. Text Augmentation
        # Load caption from report.json
        caption = ""
        try:
            with open(sample_info['report_path'], 'r') as f:
                meta = json.load(f)
                desc = meta.get('description', '')
                sentences = [s.strip() for s in desc.split('.') if s.strip()]
                if sentences:
                    # Randomly select one sentence if training and augmenting
                    caption = random.choice(sentences) if self.augment else sentences[0]
        except Exception as e:
            print(f"Error loading caption from {sample_info['report_path']}")
            print(e)
            pass  # Keep empty string on failure

        # 2. Frame Sampling (Windowed Augmentation)
        indices = self._sample_window_indices(sample_info['total_frames'])
        selected_frames = [all_frames[i] for i in indices]
        
        # 3. Load & Process PCDs
        seq_frames_processed = [] # Will hold (N, 9) arrays
        
        for _, file_path in selected_frames:
            raw_data = self._load_pcd(file_path)
            processed_data = self._process_points(raw_data)
            seq_frames_processed.append(processed_data)
        
        # Stack to (T, N, 9) numpy array
        seq_data_np = np.stack(seq_frames_processed, axis=0)

        # 4. Rotation Augmentation
        if self.augment:
            seq_data_np = self._augment_rotation(seq_data_np)

        # 5. Global Normalization
        # Normalize coords (0:3) using min/max of the *entire sequence*
        xyz = seq_data_np[:, :, 0:3]
        min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
        max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0 # Prevent div/0
        
        seq_data_np[:, :, 0:3] = (xyz - min_xyz) / range_xyz
        
        # 6. Convert to tensor - return as [T, N, 9]
        seq_data_tensor = torch.from_numpy(seq_data_np)  # [T, N, 9]

        return seq_data_tensor, caption

class ProjectionHead(nn.Module):
    """Projection head for embeddings"""
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


class MoPaTextEncoder(nn.Module):
    """Text encoder using transformer models"""
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]

class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss"""
    def __init__(self, logit_scale: float = 0.07):
        super().__init__()
        import numpy as np
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit_scale)))

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def forward(self, motion_embeds, text_embeds):
        """
        Args:
            motion_embeds: [B, D] normalized motion embeddings
            text_embeds: [B, D] normalized text embeddings
        """
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale
        logits_per_motion = logits_per_text.T

        # Symmetric loss
        caption_loss = self.contrastive_loss(logits_per_text)
        motion_loss = self.contrastive_loss(logits_per_motion)
        
        return (caption_loss + motion_loss) / 2.0



class Trainer:
    
    def __init__(self, config_path: str, resume: bool = False, experiment_name: str = None):
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        
        # Override experiment name if provided
        if experiment_name:
            self.config['experiment']['name'] = experiment_name
        
        # Setup training environment (logging, tensorboard, checkpointing)
        self.logger, self.tb_logger, self.checkpoint_manager = setup_training_environment(
            self.config, self.config['experiment']['name']
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=100)
        
        # Initialize training state
        self.current_epoch = 0
        self.best_metric = None
        self.total_steps = 0
        
        # Setup device
        self.device = torch.device(self.config['model']['pipeline_args']['device'])
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_text_encoder()
        self._setup_optimization()
        self._setup_loss()
        
        # Setup resuming if requested
        if resume or self.config.get('resume', {}).get('enabled', False):
            self._resume_training()
        
        self.logger.info("Trainer initialized successfully")
    
    def _setup_data(self):
        self.logger.info("Setting up datasets...")

        data_config = self.config['data']
        seq_config = data_config['sequence_config']
        pipe_args = self.config['model']['pipeline_args']

        self.train_dataset = HumanMLVQADataset(
            data_root=data_config['data_root'],
            seed = seq_config['seed'],
            split="train", 
            num_frames=data_config['num_frames'],
            num_points=data_config['num_points']
        )

        # TODO: Get one datasmapel for testing 
        datasample = self.train_dataset[0]
        
        self.val_dataset = HumanMLVQADataset(
            data_root=data_config['data_root'],
            seed=seq_config['seed'],
            split='test', 
            num_frames=data_config['num_frames'],
            num_points=data_config['num_points']
        )
        
        

        # # subset for quick validation
        # self.train_dataset = torch.utils.data.Subset(
        #     self.train_dataset, list(range(min(128, len(self.train_dataset))))
        # )
        
        # # small subset for quick validation
        # self.val_dataset = torch.utils.data.Subset(
        #     self.val_dataset, list(range(min(128, len(self.val_dataset))))
        # )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn
        )

        self.logger.info(f"Training dataset: {len(self.train_dataset)} sequences")
        self.logger.info(f"Validation dataset: {len(self.val_dataset)} sequences")
        self.logger.info(f"Training batches: {len(self.train_loader)}")
        self.logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def _setup_model(self):
        self.logger.info("Setting up model...")
        
        # Dynamically load pipeline module and class
        model_config = self.config['model']
        pipeline_module_name = model_config['pipeline_module']
        pipeline_class_name = model_config['pipeline_class']
        
        self.logger.info(f"Loading pipeline: {pipeline_module_name}.{pipeline_class_name}")
        
        try:
            # Import the module
            pipeline_module = importlib.import_module(pipeline_module_name)
            # Get the class
            PipelineClass = getattr(pipeline_module, pipeline_class_name)
        except ImportError as e:
            raise ImportError(f"Failed to import pipeline module '{pipeline_module_name}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Pipeline class '{pipeline_class_name}' not found in module '{pipeline_module_name}': {e}")
        
        # Get pipeline arguments from config
        pipeline_args = model_config['pipeline_args']
        
        
        # Create pipeline (it handles its own point model setup, weight loading, and freezing)
        self.model = PipelineClass(**pipeline_args)
        
        # Move model to device
        self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Pipeline class: {pipeline_class_name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_text_encoder(self):
        self.logger.info("Setting up text encoder...")
        
        text_config = self.config['model'].get('text_encoder_config', {})
        text_model_name = text_config.get('text_model_name', 'distilbert-base-uncased')
        freeze_text_encoder = text_config.get('freeze_text_encoder', False)
        text_embedding_dims = text_config.get('text_embedding_dims', 768)
        projection_dims = text_config.get('projection_dims', 256)
        dropout = text_config.get('dropout', 0.5)
        
        # Create text encoder
        self.text_encoder = MoPaTextEncoder(
            model_name=text_model_name,
            trainable=not freeze_text_encoder
        ).to(self.device)
        
        # Create text projection head
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout
        ).to(self.device)
        
        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        proj_params = sum(p.numel() for p in self.text_projection.parameters())
        
        self.logger.info(f"Text encoder: {text_params:,} params, frozen={freeze_text_encoder}")
        self.logger.info(f"Text projection: {proj_params:,} params")
    
    def _setup_optimization(self):
        self.logger.info("Setting up optimization...")
        
        opt_config = self.config['training']['optimizer']
        
        # Get separate learning rates
        motion_lr = opt_config.get('motion_lr', 1e-5)
        text_lr = opt_config.get('text_lr', 1e-5)
        head_lr = opt_config.get('head_lr', 1e-5)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Check if model has specific encoder components
        has_point_encoder = hasattr(self.model, 'point_encoder')
        has_motion_encoder = hasattr(self.model, 'motion_encoder')
        has_motion_projection = hasattr(self.model, 'motion_projection')
        
        if has_point_encoder and has_motion_encoder and has_motion_projection:
            # Model has separate components - use different learning rates
            param_groups.extend([
                {'params': self.model.point_encoder.parameters(), 'lr': motion_lr},
                {'params': self.model.motion_encoder.parameters(), 'lr': motion_lr},
                {'params': self.model.motion_projection.parameters(), 'lr': head_lr},
            ])
            self.logger.info(f"Using separate learning rates for model components")
        else:
            # Model doesn't have separate components - use all parameters
            param_groups.append({'params': self.model.parameters(), 'lr': motion_lr})
            self.logger.info(f"Using single learning rate for entire model")
        
        # Add text encoder and projection
        param_groups.extend([
            {'params': self.text_encoder.parameters(), 'lr': text_lr},
            {'params': self.text_projection.parameters(), 'lr': head_lr},
        ])
        
        self.logger.info(f"Motion encoder lr: {motion_lr}")
        self.logger.info(f"Text encoder lr: {text_lr}")
        self.logger.info(f"Projection heads lr: {head_lr}")
        
        # Create optimizer
        if opt_config['type'].lower() == 'adamw':
            self.optimizer = AdamW(
                param_groups,
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        # Create learning rate scheduler
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'].lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")
        
        self.logger.info(f"Optimizer: {opt_config['type']}")
        self.logger.info(f"Scheduler: {sched_config['type']}")
       
    def _log_trainable_parameters(self):
        """Log trainable parameter counts."""
        self.logger.info("\nTrainable Parameters:")
        self.logger.info("-" * 50)
        
        # Pipeline
        pipeline_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pipeline_total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Pipeline: {pipeline_trainable:,} / {pipeline_total:,} trainable")
        
        # Text encoder
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in self.text_encoder.parameters())
        self.logger.info(f"Text Encoder: {text_trainable:,} / {text_total:,} trainable")
        
        # Total
        self.logger.info(f"Total: {pipeline_trainable + text_trainable:,} / {pipeline_total + text_total:,} trainable")
        self.logger.info("-" * 50)
    
    def _setup_loss(self):
        self.logger.info("Setting up loss function...")
        
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'clip_contrastive':
            self.criterion = CLIPLoss(
                logit_scale=loss_config.get('logit_scale', 0.07)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']}")
        
        self.logger.info(f"Loss function: {loss_config['type']}")
        
        # Setup validation evaluator
        self.validation_evaluator = ValidationEvaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            k_values=[1, 2, 3, 5, 10],
            text_encode_fn=self._encode_text
        )
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        self.logger.info("Attempting to resume training...")
        
        resume_config = self.config.get('resume', {})
        
        resumer = TrainingResumer(self.checkpoint_manager)
        
        start_epoch, training_info = resumer.resume_training(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=resume_config.get('checkpoint_path'),
            resume_best=resume_config.get('resume_best', False),
            resume_last=resume_config.get('resume_last', True)
        )
        
        if start_epoch > 0:
            self.current_epoch = start_epoch
            self.best_metric = training_info.get('best_metric_value')
            self.logger.info(f"Resumed training from epoch {start_epoch}")
            if self.best_metric is not None:
                self.logger.info(f"Best metric so far: {self.best_metric}")
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.text_encoder.train()
        self.text_projection.train()
        
        # if self.freeze_text_encoder:
        #     self.text_encoder.eval()
        #     for param in self.text_encoder.parameters():
        #         param.requires_grad = False
        
        epoch_metrics = {
            'loss': 0.0,
            'num_batches': 0,
            'num_sequences': 0
        }
        
        start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch+1}")):
            batch_start_time = time.time()
            
            self.optimizer.zero_grad()
            
            try:
                clips, captions = batch
                clips = clips.to(self.device)
                
                temporal_features = self.model(clips)
                
                text_features = self._encode_text(captions)
                
                loss = self.criterion(temporal_features, text_features)
                
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clipping'):
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + 
                        list(self.text_encoder.parameters()) +
                        list(self.text_projection.parameters()),
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
                
                batch_size = batch[0].shape[0]
                epoch_metrics['loss'] += loss.item() * batch_size
                epoch_metrics['num_batches'] += 1
                epoch_metrics['num_sequences'] += batch_size
                
                self.metrics_tracker.update(
                    loss=loss.item(),
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
                
                if self.total_steps % 50 == 0:
                    self.tb_logger.log_scalar('train/loss_step', loss.item(), self.total_steps)
                    self.tb_logger.log_scalar('train/lr_step', self.optimizer.param_groups[0]['lr'], self.total_steps)
                
                self.tb_logger.accumulate_metric('loss', loss.item())
                self.tb_logger.accumulate_metric('learning_rate', self.optimizer.param_groups[0]['lr'])
                
                self.total_steps += 1
                
                if batch_idx % 50 == 0:
                    batch_time = time.time() - batch_start_time
                    avg_loss = self.metrics_tracker.get_moving_average('loss')
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    self.logger.info(
                        f"Epoch {self.current_epoch} | "
                        f"Batch {batch_idx}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                        f"LR: {lr:.2e} | "
                        f"Time: {batch_time:.2f}s"
                    )
            
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if epoch_metrics['num_sequences'] > 0:
            epoch_metrics['loss'] /= epoch_metrics['num_sequences']
        
        epoch_time = time.time() - start_time
        epoch_metrics['epoch_time'] = epoch_time
        
        return epoch_metrics
    
    def _encode_text(self, captions: list) -> torch.Tensor:
        """Encode text captions"""
        # Tokenize
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Encode
        text_features = self.text_encoder(input_ids, attention_mask)
        text_embeddings = self.text_projection(text_features)
        
        # Normalize
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.text_encoder.eval()
        self.text_projection.eval()
        return self.validation_evaluator.evaluate_dataloader(
            dataloader=self.val_loader,
            logger=self.logger,
            compute_global=True
        )
    
    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config['training']['num_epochs']}")
        
        num_epochs = self.config['training']['num_epochs']
        val_every_n = self.config['training']['validation']['every_n_epochs']
        val_metric = self.config['training']['validation']['metric']
        val_mode = self.config['training']['validation']['mode']
        
        best_metric = self.best_metric

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            self.logger.info(f"{'='*50}")

            train_metrics = self.train_epoch()

            self.tb_logger.log_training_metrics(
                epoch=epoch,
                loss=train_metrics['loss'],
                metrics={k: v for k, v in train_metrics.items() if k not in ['loss', 'num_batches', 'num_sequences']},
                lr=self.optimizer.param_groups[0]['lr']
            )

            self.logger.info(f"Training - Loss: {train_metrics['loss']:.4f}, Time: {train_metrics['epoch_time']:.1f}s")

            should_validate = (epoch + 1) % val_every_n == 0 or (epoch + 1) == num_epochs

            if should_validate:
                self.logger.info("Running validation...")
                val_metrics = self.validate()

                self.tb_logger.log_validation_metrics(
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics={k: v for k, v in val_metrics.items() if k not in ['loss', 'num_sequences', 'num_batches']}
                )

                for k in [1, 2, 3, 5, 10]:
                    self.tb_logger.log_scalar(f'val/recall@{k}_text_to_scene', val_metrics.get(f'recall@{k}_text_to_scene', 0), epoch)
                    self.tb_logger.log_scalar(f'val/recall@{k}_scene_to_text', val_metrics.get(f'recall@{k}_scene_to_text', 0), epoch)
                
                for k in [1, 2, 3, 5, 10]:
                    self.tb_logger.log_scalar(f'val/global_recall@{k}_text_to_scene', val_metrics.get(f'global_recall@{k}_text_to_scene', 0), epoch)
                    self.tb_logger.log_scalar(f'val/global_recall@{k}_scene_to_text', val_metrics.get(f'global_recall@{k}_scene_to_text', 0), epoch)

                self.logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}")
                self.logger.info("Batch-wise Recall@k:")
                for k in [1, 2, 3, 5, 10]:
                    self.logger.info(f"  R@{k} T→S: {100*val_metrics.get(f'recall@{k}_text_to_scene', 0):.2f}% | S→T: {100*val_metrics.get(f'recall@{k}_scene_to_text', 0):.2f}%")
                
                self.logger.info("Global Recall@k:")
                for k in [1, 2, 3, 5, 10]:
                    self.logger.info(f"  R@{k} T→S: {100*val_metrics.get(f'global_recall@{k}_text_to_scene', 0):.2f}% | S→T: {100*val_metrics.get(f'global_recall@{k}_scene_to_text', 0):.2f}%")

                current_metric = val_metrics['recall@1_text_to_scene']
                is_best = False

                if best_metric is None or current_metric > best_metric:
                    is_best = True
                    best_metric = current_metric

                if is_best:
                    self.logger.info(f"New best model! Recall@1 T→S: {100*current_metric:.2f}%")
                    
                    checkpoint_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'text_encoder_state_dict': self.text_encoder.state_dict(),
                        'text_projection_state_dict': self.text_projection.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_metric': self.best_metric,
                        'total_steps': self.total_steps,
                        'config': self.config
                    }
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        state_dict=checkpoint_state,
                        epoch=epoch,
                        metric_value=current_metric,
                        is_best=True
                    )
                    self.logger.info(f"Best model saved: {checkpoint_path}")

            if (epoch + 1) == num_epochs:
                state_dict = create_training_state_dict(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    additional_state={'total_steps': self.total_steps}
                )
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    state_dict=state_dict,
                    epoch=epoch,
                    is_best=False
                )
                self.logger.info(f"Last model saved: {checkpoint_path}")

            if isinstance(self.scheduler, ReduceLROnPlateau):
                if should_validate:
                    self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()

            self.tb_logger.log_accumulated_metrics(epoch, prefix="train")

        self.logger.info("Training completed!")
        self.logger.info(f"Best Recall@1 T→S: {100*best_metric:.2f}%")
        self.tb_logger.close()


def main():
    parser = argparse.ArgumentParser(description='Train 4D Text Grounding Pipeline')
    parser.add_argument('--config', type=str, default='configs/humanml/humanml_mopa_config_vqa.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print(f"Please create a configuration file or use the default: configs/humanml_qf_config.yaml")
        sys.exit(1)
    
    try:
        # Create trainer
        trainer = Trainer(
            config_path=args.config,
            resume=args.resume,
            experiment_name=args.experiment_name
        )
        
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()