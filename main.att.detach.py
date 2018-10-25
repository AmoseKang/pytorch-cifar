'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


from modefied_models.attention_net import Attnet1 as Attnet
from utils import progress_bar

from tensorboardX import SummaryWriter
import numpy as np

run_label = 'res18_100_2_att'

writer = SummaryWriter(log_dir=os.path.join('runs', run_label))

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', '-e', default=3, type=int, help='batch')
parser.add_argument('--batch-size', '-bs', default=480,
                    type=int, help='batch size')
# 480 for res18
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--lr_steps', default=[80, 160], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
args = parser.parse_args()


def main(att_penalty=0.1, LD=5e-4):
    att_lr = 0.1
    global best_acc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='~/Documents/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(
        root='~/Documents/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model
    print('==> Building model..')
    # from modefied_models.vgg import VGG
    # net = VGG('VGG19', num_class=100)
    from modefied_models.resnet import ResNet18
    net = ResNet18(num_classes=100)
    att = Attnet(512)
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    net = net.to(device)
    att = att.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/%s/%s' %
                                (run_label, run_label)+'.checkpoint')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss(reduction='none')
    att_loss = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    att_optimizer = optim.SGD(att.parameters(), lr=att_lr,
                              momentum=LD, weight_decay=5e-4)

    def adjust_learning_rate(optimizer, lr, epoch, lr_steps, dr=0.1):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        decay = dr ** (sum(epoch >= np.array(lr_steps)))
        lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            att_optimizer.zero_grad()

            outputs, feats = net(inputs)
            weight, aweight = att(feats)
            loss = criterion(outputs, targets)
            loss_o = loss

            # weighted_loss = (loss_o.detach() * weight.squeeze()).sum() + \
            #     weight.std() * att_penalty
            weighted_loss = (loss_o.detach() * weight.squeeze()).sum() + \
                (1. - aweight.mean()) * att_penalty
            weighted_loss.backward()
            att_optimizer.step()
            # weight, aweight = att(feats)  # reweighting
            # weighted_loss = (loss * (loss != loss.min()).float()).mean() # basic self-paced

            # loss = weighted_loss + att_penalty * (weight * weight).sum()
            loss = (loss * weight.squeeze().detach()).sum()
            # weight fix
            # if (loss > loss_o.sum()).all():
            #     loss = loss_o.mean()

            # loss = loss_o.mean()  # back to original
            loss.backward()

            optimizer.step()

            train_loss += weighted_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f/%.3f | Acc: %.3f%% (%d/%d)'
                         % (loss.item(), loss_o.mean().item(), 100.*correct/total, correct, total))
        writer.add_scalar('loss', train_loss/(batch_idx+1), epoch)
        writer.add_scalar('accurate', 100.*correct/total, epoch)

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, feats = net(inputs)
                loss = criterion(outputs, targets).mean()

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar('test_loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('test_accurate', 100.*correct/total, epoch)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'args': args
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if not os.path.isdir('checkpoint/%s' % run_label):
                os.mkdir('checkpoint/%s' % run_label)
            torch.save(state, './checkpoint/%s/' %
                       run_label+run_label+'.checkpoint')

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args.lr_steps, 0.1)
        adjust_learning_rate(att_optimizer, att_lr, epoch, args.lr_steps, 0.5)
        train(epoch)
        test(epoch)

    return best_acc


best = 0
para = None

# softmax 1e3 5e-4 26.3
# 2e3 softmax * softmax .sum

# actual  0.01 0.0001 25.6
#         0.01 0.001  25.97
#         0.05 0.01   25.39
for att_penalty in [0.2]:
    for ld in [0.9]:
        main(att_penalty, ld)
        acc = best_acc
        print('%s %s acc: %s' % (att_penalty, ld, acc))
        if acc > best:
            para = (att_penalty, ld, acc)
            best = acc

print('best set out is : %s / %s / %s' % para)
