import os
import time
import argparse

def get_acc(method, dataset, print_freq, lr, momentum, epoch, bs, shuffle, init):
    save_dir = "{}_{}_{}".format(method, dataset, str(lr).replace('.', '_'))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    channels = 3
    if dataset.lower() == 'cifar10':
        from datasets.cifar10 import get_dataset_cifar10
        train_loader, test_loader = get_dataset_cifar10(bs, (shuffle==1))
        input_num = 32 * 32
        class_num = 10
    elif dataset.lower() == 'cifar100':
        from datasets.cifar100 import get_dataset_cifar100
        train_loader, test_loader = get_dataset_cifar100(bs, (shuffle==1))
        input_num = 32 * 32
        class_num = 100
    elif dataset.lower() == 'mnist':
        from datasets.mnist import get_dataset_mnist
        train_loader, test_loader = get_dataset_mnist(bs, (shuffle==1))
        input_num = 28 * 28
        class_num = 10
        channels = 1
    else:
        raise ValueError('Dataset Not Supported!')

    model = None
    if method.lower() == 'bdr':
        pass
    elif method.lower() == 'logistics':
        max_acc = 0
        from methods.Logistics import Logistics
        model = Logistics(input_num, class_num, lr, channels=channels, init=init)

        for epoch in range(epoch):
            loss_sum, loss_cnt = 0., 0
            sta_time = time.time()

            epoch_save_dir = os.path.join(save_dir, "{:03d}".format(epoch))
            if not os.path.exists(epoch_save_dir):
                os.mkdir(epoch_save_dir)

            for idx, data in enumerate(train_loader):
                x, y = data[0], data[1]
                loss = model(x, y)
                loss_sum += loss * x.shape[0]
                loss_cnt += x.shape[0]

                if idx % print_freq == 0:
                    print('[Epoch {}] Iters: {}/{} Loss: {}'.format(epoch, idx, len(train_loader), loss_sum / loss_cnt))

            end_time = time.time()
            print('Logis Epoch Train Time {}'.format((end_time - sta_time) / len(train_loader)))

            sta_time = time.time()
            acc_sum, acc_cnt = 0., 0
            for idx, data in enumerate(test_loader):
                x, y = data[0], data[1]

                acc = model.get_accuracy(x, y)
                acc_sum += acc * x.shape[0]
                acc_cnt += x.shape[0]

            end_time = time.time()
            print('Logis Epoch Test Time {}'.format((end_time - sta_time) / len(test_loader)))

            print('[Epoch {}] Acc: {}'.format(epoch, acc_sum / acc_cnt))
            max_acc = max(max_acc, acc_sum / acc_cnt)

        weights = (model.weights + 1) * 255
        weights[weights > 255] = 255
        weights[weights < 0] = 0
        weights = weights.transpose(1, 0)

        from visual_utils import visual_weights
        visual_weights(weights, './outputs', 'logis', channels=channels)

    elif method.lower() == 'svm':
        max_acc = 0
        import torch
        from methods.SVM import SVM_SVC, SVM_Loss

        train_cri = SVM_Loss()
        model = SVM_SVC(input_num, class_num, channels)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epoch):
            loss_sum, loss_cnt = 0., 0
            sta_time = time.time()

            for idx, data in enumerate(train_loader):
                x, y = data[0], data[1].float()

                loss = train_cri(model(x), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss * x.shape[0]
                loss_cnt += x.shape[0]

                if idx % print_freq == 0:
                    print('[Epoch {}] Iters: {}/{} Loss: {}'.format(epoch, idx, len(train_loader), loss_sum / loss_cnt))

            end_time = time.time()
            print('Logis Epoch Using {}'.format((end_time - sta_time) / len(train_loader)))

            sta_time = time.time()
            acc_sum, acc_cnt = 0., 0
            for idx, data in enumerate(test_loader):
                x, y = data[0].reshape(-1, model.input_num), data[1]
                N = y.shape[0]

                outputs = model(x)
                predicted = model.convert_from_one_hot(outputs)

                acc_cnt += N
                acc_sum += (predicted.view(-1).long() == y).sum()

            end_time = time.time()
            print('Logis Epoch Test Time {}'.format((end_time - sta_time) / len(test_loader)))

            print('[Epoch {}] Acc: {}'.format(epoch, acc_sum / acc_cnt))
            max_acc = max(max_acc, acc_sum / acc_cnt)
    else:
        raise ValueError('Method Not Supported!')

    return max_acc

if __name__ == '__main__':
    epoch = 3
    svm_accs = []
    print_freq = 50
    lr = 0.01
    momentum = 0.9
    bs = 64
    shuffle = True
    init = 'zeros'
    dataset = ['cifar10'] # , 'cifar100', 'mnist']

    acc = [get_acc('logistics', dataset_item, print_freq, lr, momentum, epoch, bs, shuffle, init) for dataset_item in dataset]
    print(acc)