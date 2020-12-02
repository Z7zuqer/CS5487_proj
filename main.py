import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='CS5487_proj')
    parser.add_argument('--method', type=str,
                        default='bdr',
                        help='Name of the Method[bdr, logistics, svm]')
    parser.add_argument('--dataset', type=str,
                        default='cifar10',
                        help='Name of the Dataset[cifar10, cifar100, minist]')
    args = parser.parse_args()

    dataset = None
    if args.dataset.lower() == 'cifar10':
        pass
    elif args.dataset.lower() == 'cifar100':
        pass
    elif args.dataset.lower() == 'minist':
        pass
    else:
        raise ValueError('Dataset Not Supported!')

    model = None
    if args.method.lower() == 'bdr':
        pass
    if args.method.lower() == 'logistics':
        pass
    if args.method.lower() == 'svm':
        pass
    else:
        raise ValueError('Method Not Supported!')

    for idx, data in enumerate(dataset):
        x, y = data[0], data[1]

# yanjiang

if __name__ == '__main__':
    main()