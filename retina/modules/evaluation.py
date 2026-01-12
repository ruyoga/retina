import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

from retina.modules.models import RetinalDiseaseClassifier

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model(checkpoint_path, device='mps'):
    checkpoint_path = Path(checkpoint_path)
    print(f"\nLoading: {checkpoint_path.name}")
    model = RetinalDiseaseClassifier.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        weights_only=False
    )
    model.eval()
    model.to(device)
    return model


root = Path().resolve().parent.parent
checkpoint_dir = root / 'retina' / 'checkpoints'

models_to_load = {
    "DenseNet121": "densenet121-epoch=24-val_f1_macro=0.8245.ckpt",
    "ResNet50": "resnet50-epoch=12-val_f1=0.9374.ckpt",
    "EfficientNet-B1": "efficientnet_b1-epoch=24-val_f1_macro=0.8427.ckpt"
}

print(f"Checkpoint directory: {checkpoint_dir}")
print(f"Directory exists: {checkpoint_dir.exists()}")

loaded_models = {}
for model_name, checkpoint_file in models_to_load.items():
    checkpoint_path = checkpoint_dir / checkpoint_file
    model = load_model(checkpoint_path, device='mps')
    loaded_models[model_name] = model

print(f"\n{'=' * 80}")
print(f"LOADED {len(loaded_models)}/{len(models_to_load)} MODELS")
print(f"{'=' * 80}\n")


def get_predictions(model, dataloader, device='mps'):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze()
            probs_disease = torch.sigmoid(logits)
            probs_healthy = 1 - probs_disease
            probs_both = torch.stack([probs_healthy, probs_disease], dim=1)
            preds = (probs_disease > 0.5).int()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs_both.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_probs, all_labels


def calculate_metrics(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'roc_auc': roc_auc_score(y_true, y_probs[:, 1]),
        'avg_precision': average_precision_score(y_true, y_probs[:, 1]),
        'confusion_matrix': cm,
    }

    healthy_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    healthy_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    healthy_f1 = 2 * (healthy_precision * healthy_recall) / (healthy_precision + healthy_recall) if (healthy_precision + healthy_recall) > 0 else 0

    disease_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    disease_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    disease_f1 = 2 * (disease_precision * disease_recall) / (disease_precision + disease_recall) if (disease_precision + disease_recall) > 0 else 0

    metrics['f1_macro'] = (healthy_f1 + disease_f1) / 2

    return metrics


def plot_confusion_matrices(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = results['metrics']['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'

        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=axes[idx],
                    xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'],
                    cbar_kws={'label': 'Count'})

        axes[idx].set_title(f'{model_name}', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_roc_curves(all_results):
    plt.figure(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(all_results)))

    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['true_labels']
        y_probs = results['probabilities'][:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = results['metrics']['roc_auc']

        plt.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})', color=colors[idx])

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pr_curves(all_results):
    plt.figure(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(all_results)))

    for idx, (model_name, results) in enumerate(all_results.items()):
        y_true = results['true_labels']
        y_probs = results['probabilities'][:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = results['metrics']['avg_precision']

        plt.plot(recall, precision, lw=2.5, label=f'{model_name} (AP = {avg_precision:.4f})', color=colors[idx])

    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, label=f'Baseline ({baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(all_results):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'ROC AUC']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(all_results)))
    model_names = list(all_results.keys())

    for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
        values = [all_results[model]['metrics'][metric_key] for model in model_names]

        bars = axes[idx].bar(model_names, values, edgecolor='black', alpha=0.8, linewidth=1.5)
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        axes[idx].set_ylabel(metric_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 1.05])
        axes[idx].set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_xticklabels(model_names, rotation=15, ha='right')

        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_confidence_distributions(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(all_results)))

    for idx, (model_name, results) in enumerate(all_results.items()):
        disease_probs = results['probabilities'][:, 1]
        axes[0].hist(disease_probs, bins=40, alpha=0.6, label=model_name,
                     edgecolor='black', color=colors[idx])

    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Disease Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Disease Probability Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    confidence_data = []
    model_labels = []
    for model_name, results in all_results.items():
        max_probs = np.max(results['probabilities'], axis=1)
        confidence_data.append(max_probs)
        model_labels.append(model_name)

    bp = axes[1].boxplot(confidence_data, labels=model_labels, patch_artist=True,
                         boxprops=dict(linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='red'))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel('Confidence Score', fontsize=12)
    axes[1].set_title('Confidence Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(model_labels, rotation=15, ha='right')

    plt.suptitle('Model Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_comparison_table(all_results):
    comparison_data = []
    for model_name, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'F1 Macro': metrics['f1_macro'],
            'Specificity': metrics['specificity'],
            'ROC AUC': metrics['roc_auc'],
            'Avg Precision': metrics['avg_precision']
        })

    df = pd.DataFrame(comparison_data)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df