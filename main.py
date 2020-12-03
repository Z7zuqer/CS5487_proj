import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='CS5487_proj')
    parser.add_argument('--method', type=str,
                        default='logistics',
                        help='Name of the Method[bdr, logistics, svm]')
    parser.add_argument('--dataset', type=str,
                        default='cifar100',
                        help='Name of the Dataset[cifar10, cifar100, mnist]')
    parser.add_argument('--print_freq', type=int,
                        default=200,
                        help='Name of the Dataset[cifar10, cifar100, minist]')
    parser.add_argument('--lr', type=float,
                        default=1e-1,
                        help='learning rate')
    parser.add_argument('--epoch', type=int,
                        default=20,
                        help='epoch')
    parser.add_argument('--bs', type=int,
                        default=32,
                        help='epoch')
    parser.add_argument('--shuffle', type=int,
                        default=1,
                        help='epoch')
    args = parser.parse_args()

    channels = 3
    if args.dataset.lower() == 'cifar10':
        from datasets.cifar10 import get_dataset_cifar10
        train_loader, test_loader = get_dataset_cifar10(args.bs, (args.shuffle==1))
        input_num = 32 * 32
        class_num = 10
    elif args.dataset.lower() == 'cifar100':
        from datasets.cifar100 import get_dataset_cifar100
        train_loader, test_loader = get_dataset_cifar100(args.bs, (args.shuffle==1))
        input_num = 32 * 32
        class_num = 100
    elif args.dataset.lower() == 'mnist':
        from datasets.mnist import get_dataset_mnist
        train_loader, test_loader = get_dataset_mnist(args.bs, (args.shuffle==1))
        input_num = 28 * 28
        class_num = 10
        channels = 1
    else:
        raise ValueError('Dataset Not Supported!')

    model = None
    if args.method.lower() == 'bdr':
        pass
    elif args.method.lower() == 'logistics':
        from methods.Logistics import Logistics
        model = Logistics(input_num, class_num, args.lr, channels=channels)
    elif args.method.lower() == 'svm':
        pass
    else:
        raise ValueError('Method Not Supported!')

    for epoch in range(args.epoch):
        loss_sum, loss_cnt = 0., 0
        for idx, data in enumerate(train_loader):
            x, y = data[0], data[1]
            loss = model(x, y)
            loss_sum += loss * x.shape[0]
            loss_cnt += x.shape[0]

            if idx % args.print_freq == 0:
                print('[Epoch {}] Iters: {}/{} Loss: {}'.format(epoch, idx, len(train_loader), loss_sum / loss_cnt))

        acc_sum, acc_cnt = 0., 0
        for idx, data in enumerate(test_loader):
            x, y = data[0], data[1]

            acc = model.get_accuracy(x, y)
            acc_sum += acc * x.shape[0]
            acc_cnt += x.shape[0]

        print('[Epoch {}] Acc: {}'.format(epoch, acc_sum / acc_cnt))



if __name__ == '__main__':
    main()