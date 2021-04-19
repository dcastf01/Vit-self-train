
import os
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision
import lightly

class Dataset(Enum):
    C10 = 1
    C100 = 2
    STL10 = 3
    IN128 = 4
    PLACES205 = 5
    
def get_dataset(dataset_name):
    try:
        return Dataset[dataset_name.upper()]
    except KeyError as e:
        raise KeyError("Unknown dataset '" + dataset_name + "'. Must be one of "
                       + ', '.join([d.name for d in Dataset]))
        
class TransformsC10:
    
    # Augmentations typically used to train on cifar-10
    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

def build_dataset(dataset, batch_size):
    collate_fn=None
    
    if dataset==Dataset.C10:
        num_classes=10
        # MoCo v2 uses SimCLR augmentations, additionally, disable blur
        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=32,
            gaussian_blur=0.,
        )
        
        train_linearclassifier_transform=TransformsC10().train_classifier_transforms
        test_transform = TransformsC10().test_transforms
        
        train_dataset = datasets.CIFAR10(root='/tmp/data/',
                                         train=True,
                                         transform=None,
                                         download=True
                                         )
        
        test_dataset = datasets.CIFAR10(root='/tmp/data/',
                                        train=False,
                                        transform=None,
                                        download=True
                                        )
        
        train_dataset=\
            lightly.data.LightlyDataset.from_torch_dataset(dataset=train_dataset,
                                                                     transform=None #we use collate_fn
                                                                     )
        train_classifier_dataset=\
            lightly.data.LightlyDataset.from_torch_dataset(dataset=train_dataset,
                                                                     transform=train_linearclassifier_transform
                                                                     )
        test_dataset=\
            lightly.data.LightlyDataset.from_torch_dataset(dataset=test_dataset,
                                                                     transform=test_transform
                                                                     )
        
        
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=True,
                                num_workers=4)
    
    train_classifier_loader =\
        torch.utils.data.DataLoader(
                                train_classifier_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=4
                                 )
        
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=4)
        
        

    return train_loader,train_classifier_loader, test_loader, num_classes