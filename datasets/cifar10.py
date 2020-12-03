import torch
import torchvision
import torchvision.transforms as transforms
# 3rd-party package

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ])

valid_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ])

def get_dataset_cifar10(bs, shuffle, download=True):
    train_data = torchvision.datasets.CIFAR10(
        root='datasets',
        train=True,
        download=download,
        transform=train_transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root='datasets',
        train=False,
        download=download,
        transform=valid_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=bs,
        shuffle=shuffle,
        pin_memory=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=bs,
        shuffle=False,
        pin_memory=True, num_workers=4)

    return [train_loader, test_loader]

