import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import argparse
import pandas as pd

from retina.modules.datamodules import RetinalDataModule
from retina.modules.models import RetinalDiseaseClassifier


class SweepConfig:

    def __init__(self, wandb_config=None):
        if wandb_config is not None:
            config = wandb_config
        else:
            config = {}

        cwd = Path().resolve()
        root = cwd
        data_path = root / 'retina' / 'data'

        self.train_image_path = data_path / 'Training_Set' / 'Training_Set' / 'Training'
        self.valid_image_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'Validation'
        self.test_image_path = data_path / 'Test_Set' / 'Test_Set' / 'Test'

        self.train_labels_path = data_path / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv'
        self.valid_labels_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'RFMiD_Validation_Labels.csv'
        self.test_labels_path = data_path / 'Test_Set' / 'Test_Set' / 'RFMiD_Testing_Labels.csv'

        self.num_classes = 46
        self.model_name = config.get('model_name', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        self.learning_rate = config.get('learning_rate', 1e-4)
        self.optimizer = config.get('optimizer', 'adamw')  # Changed default to adamw
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 1e-5)

        self.batch_size = config.get('batch_size', 4)
        self.test_batch_size = max(4, self.batch_size)
        self.num_workers = 2

        # Image dimensions
        img_height = config.get('img_height', 356)
        self.img_height = img_height
        self.img_width = int(img_height * 1.5)

        # Training settings
        self.max_epochs = config.get('max_epochs', 50)

        # Gradient accumulation for effective larger batch size
        self.accumulate_grad_batches = config.get('accumulate_grad_batches', 8)

        # Learning rate scheduler
        self.use_scheduler = config.get('use_scheduler', True)
        self.scheduler = config.get('scheduler', 'plateau')
        self.scheduler_factor = config.get('scheduler_factor', 0.1)
        self.scheduler_patience = config.get('scheduler_patience', 8)
        self.scheduler_cooldown = config.get('scheduler_cooldown', 5)
        self.step_size = config.get('step_size', 15)
        self.gamma = config.get('gamma', 0.1)

        self.augmentation_strength = config.get('augmentation_strength', 'medium')

        self.use_class_weights = config.get('use_class_weights', True)
        self.class_weights = None  # Will be computed from data

        # Auto-detect best available accelerator: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.accelerator = 'cuda'
        elif torch.backends.mps.is_available():
            self.accelerator = 'mps'
        else:
            self.accelerator = 'cpu'
        self.devices = 1

        self.seed = 42

        # Checkpointing
        self.checkpoint_dir = 'sweep_checkpoints'
        self.run_name = f"{self.model_name}_sweep"


def compute_class_weights(labels_path, num_classes=46):
    """
    Compute class weights for handling imbalanced data

    Args:
        labels_path: Path to training labels CSV
        num_classes: Number of disease classes

    Returns:
        torch.Tensor: Positive class weights for BCEWithLogitsLoss
    """
    df = pd.read_csv(labels_path)

    # Get disease columns (skip ID and Disease_Risk)
    disease_cols = df.columns[2:]

    # Count positive samples per class
    pos_counts = df[disease_cols].sum().values
    total_samples = len(df)

    # Compute weights: ratio of negative to positive samples
    # If a class has very few positives, it gets higher weight
    weights = []
    for pos_count in pos_counts:
        neg_count = total_samples - pos_count
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0  # If no positive samples, use neutral weight
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    # Optional: clip extreme weights to prevent instability
    weights = torch.clamp(weights, min=0.1, max=100.0)

    return weights


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train():
    """
    Main training function called by W&B sweep agent
    Note: wandb.init() is called automatically by the sweep agent
    """
    # Get configuration from wandb (already initialized by sweep agent)
    config = SweepConfig(wandb.config)

    # Set seeds for reproducibility
    set_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("SWEEP RUN CONFIGURATION")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Batch size: {config.batch_size}")
    print(f"Accumulate grad batches: {config.accumulate_grad_batches}")
    print(f"Effective batch size: {config.batch_size * config.accumulate_grad_batches}")
    print(f"Image size: {config.img_height} Ã— {config.img_width}")
    print(f"Dropout: {config.dropout_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Max epochs: {config.max_epochs}")
    print("=" * 70)

    if config.use_class_weights:
        print("\nComputing class weights for imbalanced data...")
        config.class_weights = compute_class_weights(
            config.train_labels_path,
            config.num_classes
        )
        print(f"Class weights computed. Range: [{config.class_weights.min():.2f}, {config.class_weights.max():.2f}]")
        print(f"Mean weight: {config.class_weights.mean():.2f}")

    print("\nInitializing data module...")
    data_module = RetinalDataModule(config)

    print("Initializing model...")
    model = RetinalDiseaseClassifier(config)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f'{config.model_name}-{{epoch:02d}}-{{val_f1_macro:.4f}}',
        monitor='val_f1_macro',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=False
    )

    early_stop_callback = EarlyStopping(
        monitor='val_f1_macro',
        patience=15,
        mode='max',
        verbose=False,
        min_delta=0.001
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    # Setup trainer
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=True,  # Use W&B logger automatically
        precision='32',
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
        accumulate_grad_batches=config.accumulate_grad_batches,  # Gradient accumulation
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )

    print("\nStarting training...")
    print("=" * 70)

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Log best results
    best_score = checkpoint_callback.best_model_score

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation F1 (macro): {best_score:.4f}")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 70)

    # Log final metrics to wandb
    wandb.log({
        'best_val_f1_macro': best_score,
        'best_epoch': trainer.current_epoch,
        'total_epochs': trainer.current_epoch + 1
    })


def main():
    """
    Main entry point for training script

    Usage:
        # As part of W&B sweep (wandb.init() called automatically):
        python -m retina.experiments.train_sweep --sweep

        # Single run with defaults:
        python -m retina.experiments.train_sweep
    """
    parser = argparse.ArgumentParser(
        description='Train retinal disease classifier with W&B sweeps'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Running as part of W&B sweep (do not call wandb.init manually)'
    )
    args = parser.parse_args()

    if args.sweep:
        print("Running as part of W&B sweep...")
        train()
    else:
        print("Running single training run with default configuration...")
        wandb.init(
            project='retinal-disease-classification',
            name='single-run',
            config={
                'model_name': 'resnet50',
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'batch_size': 4,
                'max_epochs': 50
            }
        )
        train()
        wandb.finish()


if __name__ == '__main__':
    main()