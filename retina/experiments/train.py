import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from retina.modules.config import Config
from retina.modules.datamodules import RetinalDataModule
from retina.modules.models import RetinalDiseaseClassifier


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    config = Config()

    set_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)

    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) device found")
        print(f"Using accelerator: {config.accelerator}")
    else:
        print("✗ MPS device not found, using CPU")
        config.accelerator = 'cpu'

    print("\n" + "=" * 50)
    print("Initializing Data Module")
    print("=" * 50)
    data_module = RetinalDataModule(config)

    print("\n" + "=" * 50)
    print("Initializing Model")
    print("=" * 50)
    model = RetinalDiseaseClassifier(config)
    print(f"Model: ResNet50 with {config.num_classes} output classes")

    print("\n" + "=" * 50)
    print("Configuring Callbacks")
    print("=" * 50)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f'{config.model_name}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=False,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    print("\n" + "=" * 50)
    print("Configuring Weights & Biases Logger")
    print("=" * 50)

    logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        save_dir=config.wandb_save_dir,
        log_model=config.wandb_log_model,
        mode=config.wandb_mode
    )

    logger.experiment.config.update({
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'max_epochs': config.max_epochs,
        'img_height': config.img_height,
        'img_width': config.img_width,
        'num_classes': config.num_classes,
        'optimizer': 'SGD',
        'model': 'ResNet50',
        'scheduler': 'ReduceLROnPlateau'
    })

    print(f"Weights & Biases Project: {config.wandb_project}")
    print(f"Mode: {config.wandb_mode}")
    if config.wandb_entity:
        print(f"Entity: {config.wandb_entity}")

    print("\n" + "=" * 50)
    print("Initializing Trainer")
    print("=" * 50)

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,

        max_epochs=config.max_epochs,

        callbacks=callbacks,
        logger=logger,

        precision='32',

        log_every_n_steps=10,
        enable_progress_bar=True,

        deterministic=False,

        gradient_clip_val=None,

        check_val_every_n_epoch=1,
    )

    print(f"Accelerator: {config.accelerator}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")

    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    trainer.fit(
        model=model,
        datamodule=data_module
    )

    print("\n" + "=" * 50)
    print("Testing Best Model")
    print("=" * 50)

    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path='best'
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"\nView training logs at: {logger.experiment.url}")
    print(f"Or run: wandb sync {config.wandb_save_dir}")

    wandb.finish()


if __name__ == '__main__':
    main()