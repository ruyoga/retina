sweep_config_resnet50_focal = {
    'method': 'bayes',
    'name': 'resnet50-focal-loss-sweep',
    'command': ['${env}', 'python', '-m', 'retina.experiments.train_sweep', '${args}'],
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'resnet50'
        },

        'max_epochs': {
            'value': 25
        },

        'loss_type': {
            'values': ['focal', 'adaptive_focal']
        },

        'focal_alpha': {
            'distribution': 'uniform',
            'min': 0.15,
            'max': 0.40
        },

        'focal_gamma': {
            'values': [1.0, 1.5, 2.0, 2.5, 3.0]
        },

        'batch_size': {
            'values': [8, 16, 32]
        },

        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-4
        },

        'optimizer': {
            'values': ['adam', 'adamw']
        },

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },

        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },

        'scheduler': {
            'values': ['plateau', 'cosine']
        },

        'img_height': {
            'values': [356]
        }
    },

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}

sweep_config_resnet50_quick = {
    'method': 'grid',
    'name': 'resnet50-focal-quick-test',
    'command': ['${env}', 'python', '-m', 'retina.experiments.train_sweep', '${args}'],
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {'value': 'resnet50'},
        'max_epochs': {'value': 10},  # Quick test
        'loss_type': {'values': ['focal', 'adaptive_focal']},
        'focal_alpha': {'values': [0.20, 0.25, 0.30]},
        'focal_gamma': {'values': [2.0, 3.0]},
        'batch_size': {'values': [32, 64]},
        'learning_rate': {'values': [1e-4, 5e-4]},
        'optimizer': {'value': 'adamw'},
        'dropout_rate': {'values': [0.2, 0.3]},
        'scheduler': {'value': 'plateau'},
        'img_height': {'value': 356}
    }
}


sweep_config_all_models_focal = {
    'method': 'bayes',
    'name': 'all-models-focal-loss-comprehensive',
    'command': ['${env}', 'python', '-m', 'retina.experiments.train_sweep', '${args}'],
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'values': ['resnet50', 'efficientnet_b1', 'densenet121']
        },

        'max_epochs': {'value': 25},
        'loss_type': {'values': ['focal', 'adaptive_focal']},

        'focal_alpha': {
            'distribution': 'uniform',
            'min': 0.15,
            'max': 0.40
        },

        'focal_gamma': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 3.5
        },

        'batch_size': {'values': [16, 32, 64]},

        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 1e-3
        },

        'optimizer': {'values': ['adam', 'adamw']},

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },

        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },

        'scheduler': {'values': ['plateau', 'cosine']},
        'img_height': {'values': [356]}
    },

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}

sweep_config_efficientnet_focal = {
    'method': 'bayes',
    'name': 'efficientnet-b1-focal-loss-sweep',
    'command': ['${env}', 'python', '-m', 'retina.experiments.train_sweep', '${args}'],
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'efficientnet_b1'
        },

        'max_epochs': {
            'value': 25
        },

        'loss_type': {
            'values': ['focal', 'adaptive_focal']
        },

        'focal_alpha': {
            'distribution': 'uniform',
            'min': 0.20,
            'max': 0.45
        },

        'focal_gamma': {
            'values': [1.5, 2.0, 2.5, 3.0]
        },

        'batch_size': {
            'values': [16, 32, 64]
        },

        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 3e-4
        },

        'optimizer': {
            'values': ['adam', 'adamw']
        },

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },

        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        },

        'scheduler': {
            'values': ['plateau', 'cosine']
        },

        'img_height': {
            'values': [356]
        }
    },

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}


sweep_config_densenet_focal = {
    'method': 'bayes',
    'name': 'densenet121-focal-loss-sweep',
    'command': ['${env}', 'python', '-m', 'retina.experiments.train_sweep', '${args}'],
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'densenet121'
        },

        'max_epochs': {
            'value': 25
        },

        'loss_type': {
            'values': ['focal', 'adaptive_focal']
        },

        'focal_alpha': {
            'distribution': 'uniform',
            'min': 0.15,
            'max': 0.40
        },

        'focal_gamma': {
            'values': [1.0, 1.5, 2.0, 2.5, 3.0]
        },

        'batch_size': {
            'values': [16, 32, 64]
        },

        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-4
        },

        'optimizer': {
            'values': ['sgd', 'adam', 'adamw']
        },

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },

        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.4
        },

        'scheduler': {
            'values': ['plateau', 'cosine', 'step']
        },

        'img_height': {
            'values': [356]
        }
    },

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}


def get_sweep_config(sweep_type='resnet50_focal'):
    configs = {
        'resnet50_focal': sweep_config_resnet50_focal,
        'efficientnet_focal': sweep_config_efficientnet_focal,
        'densenet_focal': sweep_config_densenet_focal,
        'resnet50_quick': sweep_config_resnet50_quick,
        'all_models': sweep_config_all_models_focal
    }

    if sweep_type not in configs:
        print(f"Unknown sweep type: {sweep_type}")
        print(f"Available types: {list(configs.keys())}")
        return sweep_config_resnet50_focal

    return configs[sweep_type]