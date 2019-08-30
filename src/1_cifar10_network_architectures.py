'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from trainer import Trainer
from metrics import accuracy
from utils import save_checkpoint
from utils import save_onnx_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_test_metric = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0) # Note, on windows, num_workers should be 0. (https://github.com/pytorch/pytorch/issues/4418)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0) # Note, on windows, num_workers should be 0. (https://github.com/pytorch/pytorch/issues/4418)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG11')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)

#if device == 'cuda':
#    net = torch.nn.DataParallel(net) # Note, that It is not possible to save as ONNX model when using DataParallel (https://github.com/pytorch/pytorch/issues/13397)
#    cudnn.benchmark = True # This enables specifc optimizations for the network architecture (said to work best for fixed sized inputs).

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Setup trainer.
criterion = nn.CrossEntropyLoss()
metric = accuracy
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
trainer = Trainer(net, criterion, metric, optimizer, device)

# Run training loop.
epochs = 50
print("Training for {0} epochs...".format(epochs))

for epoch in range(start_epoch, start_epoch + epochs):

    start = time.time()
    (train_loss, train_metric) = trainer.train(trainloader)
    (test_loss, test_metric) = trainer.evaluate(testloader)
    end = time.time()

    total_epoch_time = end - start

    print("Epoch: {0} | Total Epoch Time: {1} seconds: ".format(epoch, total_epoch_time))
    print("    - Training Loss: {0} | Test Loss: {1}".format(train_loss, test_loss))
    print("    - Training Metric: {0} | Test Metric: {1}".format(train_metric, test_metric))

    # Save checkpoint if new best model.
    if test_metric > best_test_metric:
        save_checkpoint(epoch, trainer.model, test_metric)
        best_test_metric = test_metric
        print("Saving best checkpoint: New best test score {0}".format(best_test_metric))

# Save model as onnx.
dummy_input = torch.randn(10, 3, 32, 32, device='cuda')
save_onnx_model(trainer.model, dummy_input)