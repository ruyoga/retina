import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import pandas as pd

from retina.modules.datamodules import RetinalDataModule
from retina.modules.models import RetinalDiseaseClassifier


class SweepConfig:

    def __init__(self, wandb_config=None):
        if wandb_config is not None:
            config = wandb_config
        else:
            config = {}

        root = Path.cwd()

        if (root / 'retina' / 'data').exists():
            data_path = root / 'retina' / 'data'
        elif (root / 'data').exists():
            data_path = root / 'data'
        else:
            data_path = root / 'retina' / 'data'

        self.train_image_path = data_path / 'Training_Set' / 'Training_Set' / 'Training'
        self.valid_image_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'Validation'
        self.test_image_path = data_path / 'Test_Set' / 'Test_Set' / 'Test'

        self.train_labels_path = data_path / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv'
        self.valid_labels_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'RFMiD_Validation_Labels.csv'
        self.test_labels_path = data_path / 'Test_Set' / 'Test_Set' / 'RFMiD_Testing_Labels.csv'

        self.num_classes = 1
        self.task = 'binary'

        self.model_name = config.get('model_name', 'resnet50')
        self.pretrained = True
        self.dropout_rate = config.get('dropout_rate', 0.3)

        self.loss_type = config.get('loss_type', 'focal')
        self.focal_alpha = config.get('focal_alpha', 0.25)
        self.focal_gamma = config.get('focal_gamma', 2.0)

        self.batch_size = config.get('batch_size', 32)
        self.test_batch_size = self.batch_size
        self.num_workers = 2
        self.max_epochs = config.get('max_epochs', 25)

        self.optimizer = config.get('optimizer', 'adamw')
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-3)
        self.momentum = config.get('momentum', 0.9)

        img_height = config.get('img_height', 356)
        self.img_height = img_height
        self.img_width = int(img_height * 1.5)

        self.use_scheduler = True
        self.scheduler = config.get('scheduler', 'plateau')
        self.scheduler_factor = 0.1
        self.scheduler_patience = 5
        self.scheduler_cooldown = 3

        if torch.cuda.is_available():
            self.accelerator = 'cuda'
        elif torch.backends.mps.is_available():
            self.accelerator = 'mps'
        else:
            self.accelerator = 'cpu'
        self.devices = 1

        self.checkpoint_dir = 'sweep_checkpoints_focal'

        self.seed = 69


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def train():
    torch.set_float32_matmul_precision('medium')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("✓ Cleared CUDA cache at start of run")

    config = SweepConfig(wandb.config)

    set_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)


    model = None
    trainer = None
    data_module = None

    try:
        data_module = RetinalDataModule(config)
        model = RetinalDiseaseClassifier(config)

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
            patience=5,
            mode='max',
            verbose=True,
            min_delta=0.001
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

        wandb_logger = WandbLogger(
            project=wandb.run.project if wandb.run else 'retina-binary-focal-sweep',
            log_model=False,
            save_dir='wandb_logs'
        )

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=config.devices,
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            precision='32',
            log_every_n_steps=10,
            enable_progress_bar=True,
            deterministic=False,
            check_val_every_n_epoch=1,
            enable_model_summary=False,
            gradient_clip_val=1.0
        )

        trainer.fit(model=model, datamodule=data_module)
        best_f1_macro = checkpoint_callback.best_model_score

        return

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        print("\nCleaning up memory...")

        if model is not None:
            del model
        if trainer is not None:
            del trainer
        if data_module is not None:
            del data_module

        import gc
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3

def main():
    if wandb.run is None:
        wandb.init(
            project='retina-binary-focal-sweep',
            name='test-run',
            config={
                'model_name': 'resnet50',
                'loss_type': 'focal',
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'weight_decay': 1e-3,
                'dropout_rate': 0.3,
                'scheduler': 'plateau',
                'img_height': 356,
                'max_epochs': 10
            }
        )
        train()
        wandb.finish()
    else:
        train()


if __name__ == '__main__':
    main()