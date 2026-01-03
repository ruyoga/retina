import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class RetinalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, -1]

        label_array = self.df.iloc[idx, 1:-3].values.astype("float32")

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        labels = torch.from_numpy(label_array)
        return image, labels


class RetinalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_files = self._get_file_list(self.config.train_image_path)
        valid_files = self._get_file_list(self.config.valid_image_path)
        test_files = self._get_file_list(self.config.test_image_path)

        train_labels = pd.read_csv(self.config.train_labels_path)
        valid_labels = pd.read_csv(self.config.valid_labels_path)
        test_labels = pd.read_csv(self.config.test_labels_path)

        train_files_df = self._create_files_df(train_files)
        valid_files_df = self._create_files_df(valid_files)
        test_files_df = self._create_files_df(test_files)

        self.train_df = pd.merge(train_labels, train_files_df, left_on='ID', right_on='ids')
        self.valid_df = pd.merge(valid_labels, valid_files_df, left_on='ID', right_on='ids')
        self.test_df = pd.merge(test_labels, test_files_df, left_on='ID', right_on='ids')

        self.train_df['full_file_paths'] = str(self.config.train_image_path) + '/' + self.train_df['filenames']
        self.valid_df['full_file_paths'] = str(self.config.valid_image_path) + '/' + self.valid_df['filenames']
        self.test_df['full_file_paths'] = str(self.config.test_image_path) + '/' + self.test_df['filenames']

    def _get_file_list(self, path):
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def _create_files_df(self, file_list):
        ids = [f.split('.')[0] for f in file_list]
        df = pd.DataFrame({
            'ids': pd.Series(ids, dtype='int64'),
            'filenames': file_list
        })
        return df

    def _get_train_transforms(self):
        return A.Compose([
            A.Resize(self.config.img_height, self.config.img_width),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _get_test_transforms(self):
        return A.Compose([
            A.Resize(self.config.img_height, self.config.img_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def train_dataloader(self):
        train_dataset = RetinalDataset(
            df=self.train_df,
            transform=self._get_train_transforms()
        )
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def val_dataloader(self):
        valid_dataset = RetinalDataset(
            df=self.valid_df,
            transform=self._get_test_transforms()
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def test_dataloader(self):
        test_dataset = RetinalDataset(
            df=self.test_df,
            transform=self._get_test_transforms()
        )
        return DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )