import torchvision
# 3rd-party package

def get_dataset_minist(download=True):
    minist = torchvision.datasets.MNIST(
        root='datasets',
        train=True,
        download=download
    )

    minist_test = torchvision.datasets.MNIST(
        root='datasets',
        train=False,
        download=download
    )
    return [minist, minist_test]