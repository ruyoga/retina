import wandb
from pathlib import Path

# Load sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'values': ['resnet50', 'efficientnet_b1', 'densenet121']
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'optimizer': {
            'values': ['sgd', 'adam', 'adamw']
        },
        'batch_size': {
            'values': [2, 4, 8]
        },
        'img_height': {
            'values': [356, 512, 712]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'scheduler': {
            'values': ['plateau', 'cosine', 'step']
        },
        'max_epochs': {
            'value': 50
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(
    sweep_config,
    project='retinal-disease-classification'
)

print(f"Sweep ID: {sweep_id}")
print(f"Run with: wandb agent {sweep_id}")