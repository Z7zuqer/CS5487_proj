import torchvision
# 3rd-party package

def get_dataset_cifar100(download=True):
    cifar100 = torchvision.datasets.CIFAR100(
        root='datasets',
        train=True,
        download=download
    )

    cifar100_test = torchvision.datasets.CIFAR100(
        root='datasets',
        train=False,
        download=download
    )
    return [cifar100, cifar100_test]