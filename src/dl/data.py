"""PyTorch dataset and data loaders for Fashion-MNIST."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from aml.data import get_train_val_test_splits


class FashionMNISTDataset(Dataset):
    """Wrap Fashion-MNIST arrays as image and label tensors."""

    def __init__(self, images, labels, transform=None):
        # CNNs expect one channel: (N, 1, 28, 28), not (N, 28, 28).
        self.images = torch.from_numpy(np.asarray(images)).float().unsqueeze(1)
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Augmentation is applied only when a transform was given (training set).
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(batch_size=128, data_dir="data/raw", seed=42, num_workers=0):
    """Return train, validation, and test DataLoaders in that order.

    The training loader applies random flips and small rotations. Validation
    and test images are left unchanged so their scores are comparable.
    """
    # Use a fixed seed so shuffling and random image transforms can be repeated.
    torch.manual_seed(seed)

    data = get_train_val_test_splits(data_dir=data_dir, seed=seed)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])

    train_dataset = FashionMNISTDataset(
        data["X_train"], data["y_train"], transform=train_transform
    )
    val_dataset = FashionMNISTDataset(data["X_val"], data["y_val"])
    test_dataset = FashionMNISTDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
