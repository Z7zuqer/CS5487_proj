import torch
import torchvision
import torchvision.transforms as transforms
# 3rd-party package

train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.3087, )),
                    ])


def get_dataset_mnist(bs, shuffle, download=True):
    train_data = torchvision.datasets.MNIST(
        root='datasets',
        train=True,
        download=download,
        transform=train_transform
    )

    test_data = torchvision.datasets.MNIST(
        root='datasets',
        train=False,
        download=download,
        transform=train_transform
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

