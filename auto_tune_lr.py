import os

from get_accs import get_acc

epoch = 10
svm_accs = []
print_freq = 50
lr_choice = [0, 0.1, 1e-2, 1e-4, 0.5]
momentum = 0.9
bs = 64
shuffle = True

logistics_cifar10_accs = []
logistics_cifar100_accs = []
logistics_mnist_accs = []

svm_cifar10_accs = []
svm_cifar100_accs = []
svm_mnist_accs = []

for lr in lr_choice:
    acc = get_acc('logistics', 'cifar10', print_freq, lr, momentum, epoch, bs, shuffle)
    logistics_cifar10_accs.append(acc)
    print('logistics cifar10 lr: {} acc: {}'.format(lr, acc))

    acc = get_acc('logistics', 'cifar100', print_freq, lr, momentum, epoch, bs, shuffle)
    logistics_cifar100_accs.append(acc)
    print('logistics cifar100 lr: {} acc: {}'.format(lr, acc))

    acc = get_acc('logistics', 'mnist', print_freq, lr, momentum, epoch, bs, shuffle)
    logistics_mnist_accs.append(acc)
    print('logistics mnist lr: {} acc: {}'.format(lr, acc))

    acc = get_acc('svm', 'cifar10', print_freq, lr, momentum, epoch, bs, shuffle)
    svm_cifar10_accs.append(acc)
    print('svm cifar10 lr: {} acc: {}'.format(lr, acc))

    acc = get_acc('svm', 'cifar100', print_freq, lr, momentum, epoch, bs, shuffle)
    svm_cifar100_accs.append(acc)
    print('svm cifar100 lr: {} acc: {}'.format(lr, acc))

    acc = get_acc('svm', 'mnist', print_freq, lr, momentum, epoch, bs, shuffle)
    svm_mnist_accs.append(acc)
    print('svm mnist lr: {} acc: {}'.format(lr, acc))


