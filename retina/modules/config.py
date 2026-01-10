from pathlib import Path
import torch


class Config:
    cwd = Path().resolve()
    root = cwd
    data_path = root / 'retina' / 'data'

    train_image_path = data_path / 'Training_Set' / 'Training_Set' / 'Training'
    valid_image_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'Validation'
    test_image_path = data_path / 'Test_Set' / 'Test_Set' / 'Test'

    train_labels_path = data_path / 'Training_Set' / 'Training_Set' / 'RFMiD_Training_Labels.csv'
    valid_labels_path = data_path / 'Evaluation_Set' / 'Evaluation_Set' / 'RFMiD_Validation_Labels.csv'
    test_labels_path = data_path / 'Test_Set' / 'Test_Set' / 'RFMiD_Testing_Labels.csv'

    num_classes = 46
    img_height = 356
    img_width = img_height * 1.5

    batch_size = 16
    test_batch_size = 16
    num_workers = 2
    max_epochs = 50
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 0.0005
    model_name = 'resnet50'

    scheduler_factor = 0.1
    scheduler_patience = 8
    scheduler_cooldown = 10

    seed = 69

    # Auto-detect best available accelerator: CUDA > MPS > CPU
    if torch.cuda.is_available():
        accelerator = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
    else:
        accelerator = 'cpu'
    devices = 1

    checkpoint_dir = 'checkpoints'

    wandb_project = 'retina'
    wandb_entity = None
    wandb_run_name = None
    wandb_mode = 'online'
    wandb_save_dir = 'wandb_logs'
    wandb_log_model = True