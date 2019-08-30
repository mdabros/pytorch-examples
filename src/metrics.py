
'''Metric functions for PyTorch'''
import torch
import torch.nn as nn
import torch.nn.init as init

def accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = outputs.size(0)
    correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy
