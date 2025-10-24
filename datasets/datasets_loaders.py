import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Optional, Tuple
import os

# Classes in this file are yet to be tested

class NCSNDataset:
    """Base class for NCSN datasets"""
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Returns train and test dataloaders."""
        raise NotImplementedError


class MNISTDataset(NCSNDataset):
    """
    MNIST dataset for NCSN.
    Images are rescaled to [0, 1] (as specified in the paper).
    """
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        super().__init__(root, batch_size, num_workers, pin_memory)
        
        # Rescale so that pixel values are in [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
        ])
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader


class CelebADataset(NCSNDataset):
    """
    CelebA dataset for NCSN. Configured as specified in the paper (images are first center-cropped to 140 x 140 
    and then resized to 32 x 32, images are rescaled to [0, 1])
    """
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        img_size: int = 32
    ):
        super().__init__(root, batch_size, num_workers, pin_memory)
        self.img_size = img_size
        
        # Training transform with random flip
        self.train_transform = transforms.Compose([
            transforms.CenterCrop(140),  
            transforms.Resize(img_size),  
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),  
        ])
        
        # Test transform without flip
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = datasets.CelebA(
            root=self.root,
            split='train',
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CelebA(
            root=self.root,
            split='test',
            download=True,
            transform=self.test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader


class CIFAR10Dataset(NCSNDataset):
    """
    CIFAR-10 dataset for NCSN. Configured as specified in the paper (Images are rescaled to [0, 1], and during training, we randomly flip the images)
    """
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        super().__init__(root, batch_size, num_workers, pin_memory)
        
        # Training transform with random flip 
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),  
        ])
        
        # Test transform without flip
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader


def get_dataset(
    dataset_name: str,
    root: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get dataset loaders.
    
    Args:
        dataset_name: One of 'mnist', 'celeba', 'cifar10'
        root: Root directory for datasets
        batch_size: Batch size (paper uses 128)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        dataset = MNISTDataset(root, batch_size, num_workers, pin_memory)
    elif dataset_name == 'celeba':
        dataset = CelebADataset(root, batch_size, num_workers, pin_memory, **kwargs)
    elif dataset_name == 'cifar10':
        dataset = CIFAR10Dataset(root, batch_size, num_workers, pin_memory)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: mnist, celeba, cifar10")
    
    return dataset.get_loaders()