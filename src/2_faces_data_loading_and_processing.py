'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from torch.utils.data import DataLoader
from models import *
from models import faces
from trainer import Trainer
from utils import *
from face_landmarks_dataset import FaceLandmarksDataset

parser = argparse.ArgumentParser(description='PyTorch FACE Landmarks Training')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seeds_for_reproducible_mode(seed=113, device=device)


# Data
print('==> Preparing data..')
trainTransformed = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           augment_data=True)
testTransformed = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           augment_data=True)

trainloader = DataLoader(trainTransformed, batch_size=4, shuffle=True, num_workers=0) # Note, on windows, num_workers should be 0. (https://github.com/pytorch/pytorch/issues/4418)
# note using training set as test set.
testloader = DataLoader(testTransformed, batch_size=4, shuffle=False, num_workers=0) # Note, on windows, num_workers should be 0. (https://github.com/pytorch/pytorch/issues/4418)

# Model
print('==> Building model..')
net = faces.CustomFaces(output_count=136)
net = net.to(device)

best_test_metric = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Setup trainer.
criterion = nn.MSELoss()
metric = nn.MSELoss()
optimizer = optim.Adam(net.parameters()) #optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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

# Load best model.
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

# Save best model as onnx.
print("Saving best model in ONNX format...")
dummy_input = torch.randn(10, 3, 32, 32, device='cuda')
save_onnx_model(net, dummy_input)