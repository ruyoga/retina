import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall


class RetinalDiseaseClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        model_name = getattr(config, 'model_name', 'resnet50')
        self.model_type = config.model_name

        self.model = self._build_model(config.model_name)

        # Setup loss function with optional class weights for imbalanced data
        pos_weight = getattr(config, 'class_weights', None)
        if pos_weight is not None:
            # Move weights to correct device
            self.register_buffer('pos_weight', pos_weight)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='mean')
            print(f"✓ Using class weights for imbalanced data (range: {pos_weight.min():.2f}-{pos_weight.max():.2f})")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
            print("⚠ WARNING: Not using class weights - may struggle with imbalanced data")

        self._setup_metrics()

    def _build_model(self, model_name):
        num_classes = self.config.num_classes
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
        # Use new weights API instead of deprecated pretrained parameter
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
        # Use new weights API instead of deprecated pretrained parameter
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
        num_classes = self.config.num_classes

        self.train_f1_macro = MultilabelF1Score(num_labels=num_classes, average='macro')
        self.train_f1_micro = MultilabelF1Score(num_labels=num_classes, average='micro')
        self.val_f1_macro = MultilabelF1Score(num_labels=num_classes, average='macro')
        self.val_f1_micro = MultilabelF1Score(num_labels=num_classes, average='micro')

        self.val_precision = MultilabelPrecision(num_labels=num_classes, average='macro')
        self.val_recall = MultilabelRecall(num_labels=num_classes, average='macro')
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Get predictions for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        # Update metrics
        self.train_f1_macro(preds, labels.int())
        self.train_f1_micro(preds, labels.int())

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_macro', self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_micro', self.train_f1_micro, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Get predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        # Update metrics
        self.val_f1_macro(preds, labels.int())
        self.val_f1_micro(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_auroc(probs, labels.int())

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_macro', self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_micro', self.val_f1_micro, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.val_f1_macro(preds, labels.int())
        self.val_f1_micro(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_auroc(probs, labels.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.val_f1_macro, on_step=False, on_epoch=True)
        self.log('test_f1_micro', self.val_f1_micro, on_step=False, on_epoch=True)
        self.log('test_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('test_auroc', self.val_auroc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_name = getattr(self.config, 'optimizer', 'sgd').lower()
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
                patience=getattr(self.config, 'scheduler_patience', 8),
                cooldown=getattr(self.config, 'scheduler_cooldown', 10),
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

        logits = self(images)
        probs = torch.sigmoid(logits)
        return probs


def get_model_info(model_name='resnet50'):
    info = {
        'resnet50': {
            'name': 'ResNet50',
            'parameters': '25.6M',
        },
        'efficientnet_b1': {
            'name': 'EfficientNet-B1',
            'parameters': '7.8M',
        },
        'densenet121': {
            'name': 'DenseNet121',
            'parameters': '8.0M',
        }
    }

    return info.get(model_name, {})


def list_available_models():
    return ['resnet50', 'efficientnet_b1', 'densenet121']