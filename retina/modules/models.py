import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryConfusionMatrix
)

from retina.modules.losses import FocalLoss, AdaptiveFocalLoss


class RetinalDiseaseClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model_type = config.model_name
        self.task = getattr(config, 'task', 'binary')

        self.model = self._build_model(config.model_name)

        loss_type = getattr(config, 'loss_type', 'focal')
        focal_alpha = getattr(config, 'focal_alpha', 0.25)
        focal_gamma = getattr(config, 'focal_gamma', 2.0)

        if loss_type == 'adaptive_focal':
            self.criterion = AdaptiveFocalLoss(
                gamma=focal_gamma,
                reduction='mean'
            )
        else:
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean'
            )

        self._setup_metrics()

    def _build_model(self, model_name):
        num_classes = 1
        pretrained = getattr(self.config, 'pretrained', True)
        dropout_rate = getattr(self.config, 'dropout_rate', 0.0)

        if model_name == 'resnet50':
            return self._build_resnet50(num_classes, pretrained, dropout_rate)
        elif model_name == 'efficientnet_b1':
            return self._build_efficientnet_b1(num_classes, pretrained, dropout_rate)
        elif model_name == 'densenet121':
            return self._build_densenet121(num_classes, pretrained, dropout_rate)
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from: resnet50, efficientnet_b1, densenet121")

    def _build_resnet50(self, num_classes, pretrained, dropout_rate):
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features

        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            model.fc = nn.Linear(num_features, num_classes)

        return model

    def _build_efficientnet_b1(self, num_classes, pretrained, dropout_rate):
        if pretrained:
            weights = models.EfficientNet_B1_Weights.DEFAULT
        else:
            weights = None

        model = models.efficientnet_b1(weights=weights)
        num_features = model.classifier[1].in_features

        if dropout_rate > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(num_features, num_classes)
            )
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            )

        return model

    def _build_densenet121(self, num_classes, pretrained, dropout_rate):
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        model = models.densenet121(weights=weights)
        num_features = model.classifier.in_features

        if dropout_rate > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            model.classifier = nn.Linear(num_features, num_classes)

        return model

    def _setup_metrics(self):
        self.train_f1 = BinaryF1Score()
        self.train_acc = BinaryAccuracy()

        self.val_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_auroc = BinaryAUROC()
        self.val_confusion = BinaryConfusionMatrix()

        self.test_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_auroc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze()
        labels = labels.squeeze()

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.train_f1(preds, labels.int())
        self.train_acc(preds, labels.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze()
        labels = labels.squeeze()

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.val_f1(preds, labels.int())
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_auroc(probs, labels.int())
        self.val_confusion(preds, labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        conf_matrix = self.val_confusion.compute()

        tn, fp, fn, tp = conf_matrix.ravel()

        healthy_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        healthy_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        healthy_f1 = 2 * (healthy_precision * healthy_recall) / (healthy_precision + healthy_recall) if (healthy_precision + healthy_recall) > 0 else 0

        disease_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        disease_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        disease_f1 = 2 * (disease_precision * disease_recall) / (disease_precision + disease_recall) if (disease_precision + disease_recall) > 0 else 0

        macro_f1 = (healthy_f1 + disease_f1) / 2

        self.log('val_f1_macro', macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_healthy', healthy_f1, on_step=False, on_epoch=True)
        self.log('val_f1_disease', disease_f1, on_step=False, on_epoch=True)

        total = tn + fp + fn + tp
        pred_disease_rate = (tp + fp) / total if total > 0 else 0

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze()
        labels = labels.squeeze()

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.test_f1(preds, labels.int())
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_auroc(probs, labels.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_name = getattr(self.config, 'optimizer', 'adamw').lower()
        lr = self.config.learning_rate
        weight_decay = getattr(self.config, 'weight_decay', 0.0005)

        if optimizer_name == 'sgd':
            momentum = getattr(self.config, 'momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        use_scheduler = getattr(self.config, 'use_scheduler', True)

        if not use_scheduler:
            return optimizer

        scheduler_name = getattr(self.config, 'scheduler', 'plateau').lower()

        if scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=getattr(self.config, 'scheduler_factor', 0.1),
                patience=getattr(self.config, 'scheduler_patience', 5),
                cooldown=getattr(self.config, 'scheduler_cooldown', 3),
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
        elif scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=lr * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(self.config, 'step_size', 10),
                gamma=getattr(self.config, 'gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        logits = self(images).squeeze()
        probs = torch.sigmoid(logits)
        return probs