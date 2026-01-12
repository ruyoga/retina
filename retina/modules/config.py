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

    num_classes = 1
    task = 'binary'

    img_height = 356
    img_width = int(img_height * 1.5)  # 534
    focal_alpha = 0.25
    focal_gamma = 2.0

    batch_size = 64
    test_batch_size = 64
    num_workers = 2
    max_epochs = 30

    optimizer = 'adamw'
    learning_rate = 0.0001
    weight_decay = 0.01
    momentum = 0.9

    model_name = 'resnet50'
    pretrained = True
    dropout_rate = 0.3

    use_scheduler = True
    scheduler = 'plateau'
    scheduler_factor = 0.1
    scheduler_patience = 5
    scheduler_cooldown = 3

    seed = 69

    if torch.cuda.is_available():
        accelerator = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
    else:
        accelerator = 'cpu'
    devices = 1

    checkpoint_dir = 'checkpoints_binary_focal'

    wandb_project = 'retina-binary-focal'
    wandb_entity = None
    wandb_run_name = 'focal_loss_baseline'
    wandb_mode = 'online'
    wandb_save_dir = 'wandb_logs_binary_focal'
    wandb_log_model = True