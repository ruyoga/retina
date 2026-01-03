import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score


class RetinalDiseaseClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = self._build_model()

        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

        self.train_f1_macro = MultilabelF1Score(
            num_labels=config.num_classes,
            average='macro'
        )
        self.train_f1_micro = MultilabelF1Score(
            num_labels=config.num_classes,
            average='micro'
        )
        self.val_f1_macro = MultilabelF1Score(
            num_labels=config.num_classes,
            average='macro'
        )
        self.val_f1_micro = MultilabelF1Score(
            num_labels=config.num_classes,
            average='micro'
        )

    def _build_model(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, self.config.num_classes)

        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.train_f1_macro(preds, labels.int())
        self.train_f1_micro(preds, labels.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_macro', self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_micro', self.train_f1_micro, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.val_f1_macro(preds, labels.int())
        self.val_f1_micro(preds, labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_macro', self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_micro', self.val_f1_micro, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.val_f1_macro(preds, labels.int())
        self.val_f1_micro(preds, labels.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.val_f1_macro, on_step=False, on_epoch=True)
        self.log('test_f1_micro', self.val_f1_micro, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Minimize validation loss
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            cooldown=self.config.scheduler_cooldown,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        logits = self(images)
        probs = torch.sigmoid(logits)
        return probs