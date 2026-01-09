sweep_config_comprehensive = {
    'method': 'bayes',
    'name': 'retinal-disease-comprehensive-sweep',
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

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },

        'batch_size': {
            'values': [8, 16, 32]
        },

        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },

        'img_height': {
            'values': [512, 712, 1024]
        },

        'scheduler': {
            'values': ['plateau', 'cosine', 'step']
        },

        'augmentation_strength': {
            'values': ['light', 'medium', 'heavy']
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}

sweep_config_quick = {
    'method': 'bayes',
    'name': 'retinal-disease-quick-sweep',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'values': ['resnet50', 'efficientnet_b1', 'densenet121']
        },

        'learning_rate': {
            'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        },

        'optimizer': {
            'values': ['sgd', 'adam']
        },

        'batch_size': {
            'values': [8, 16]
        },

        'dropout_rate': {
            'values': [0.0, 0.2, 0.5]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        's': 2
    }
}


sweep_config_resnet = {
    'method': 'bayes',
    'name': 'retinal-disease-resnet50-sweep',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'resnet50'  # Fixed
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'optimizer': {
            'values': ['sgd', 'adam', 'adamw']
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.85,
            'max': 0.95
        }
    }
}

sweep_config_efficientnet = {
    'method': 'bayes',
    'name': 'retinal-disease-efficientnet-sweep',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'efficientnet_b1'
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
            'max': 1e-3
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        }
    }
}

sweep_config_densenet = {
    'method': 'bayes',
    'name': 'retinal-disease-densenet-sweep',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'value': 'densenet121'  # Fixed
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'optimizer': {
            'values': ['sgd', 'adam', 'adamw']
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.4
        }
    }
}

sweep_config_grid = {
    'method': 'grid',
    'name': 'retinal-disease-grid-search',
    'metric': {
        'name': 'val_f1_macro',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'values': ['resnet50', 'efficientnet_b1', 'densenet121']
        },
        'learning_rate': {
            'values': [1e-4, 5e-4, 1e-3]
        },
        'optimizer': {
            'values': ['sgd', 'adam']
        },
        'batch_size': {
            'values': [8, 16]
        },
        'dropout_rate': {
            'values': [0.0, 0.3]
        }
    }
}

sweep_config_random = {
    'method': 'random',
    'name': 'retinal-disease-random-search',
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
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        }
    }
}


def get_sweep_config(sweep_type='quick'):
    configs = {
        'comprehensive': sweep_config_comprehensive,
        'quick': sweep_config_quick,
        'resnet': sweep_config_resnet,
        'efficientnet': sweep_config_efficientnet,
        'densenet': sweep_config_densenet,
        'grid': sweep_config_grid,
        'random': sweep_config_random
    }

    return configs.get(sweep_type, sweep_config_quick)