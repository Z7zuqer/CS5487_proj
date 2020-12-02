import torchvision
# 3rd-party package

def get_dataset_cifar10(download=True):
    cifar10 = torchvision.datasets.CIFAR10(
        root='datasets',
        train=True,
        download=download
    )

    cifar10_test = torchvision.datasets.CIFAR10(
        root='datasets',
        train=False,
        download=download
    )
    return [cifar10, cifar10_test]

