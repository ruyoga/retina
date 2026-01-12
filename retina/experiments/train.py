import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import pandas as pd

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
    torch.set_float32_matmul_precision('medium')

    config = Config()

    set_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)

    if torch.cuda.is_available():
        print(f"CUDA device found: {torch.cuda.get_device_name(0)}")
        print(f"  Using accelerator: {config.accelerator}")
    elif torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) device found")
        print(f"  Using accelerator: {config.accelerator}")
    else:
        print("No GPU acceleration available, using CPU")
        print(f"  Using accelerator: {config.accelerator}")

    data_module = RetinalDataModule(config)

    model = RetinalDiseaseClassifier(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f'{config.model_name}-{{epoch:02d}}-{{val_f1:.4f}}',
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=10,
        mode='max',
        verbose=True,
        min_delta=0.005
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        save_dir=config.wandb_save_dir,
        log_model=config.wandb_log_model,
        mode=config.wandb_mode
    )

    logger.experiment.config.update({
        'task': 'binary_classification',
        'loss_function': 'focal_loss',
        'focal_alpha': config.focal_alpha,
        'focal_gamma': config.focal_gamma,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'max_epochs': config.max_epochs,
        'img_height': config.img_height,
        'img_width': config.img_width,
        'optimizer': config.optimizer,
        'model': config.model_name,
        'scheduler': config.scheduler,
        'accelerator': config.accelerator,
        'dropout': config.dropout_rate,
    })

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
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )

    torch.serialization.add_safe_globals([Config])

    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path='best'
    )

    print("Training Complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation F1: {checkpoint_callback.best_model_score}")

    wandb.finish()


if __name__ == '__main__':
    main()
