'''Simple Trainer class for PyTorch.'''
import torch

class Trainer(object):
    def __init__(self, model, criterion, optimizer, device):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.to(device)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

    def train(self, train_loader):
        '''
            Trains the model for a single epoch
        '''

        loss_sum = 0.0
        N = 0
        self.model.train()
        outputs_data = list()
        targets_data = list()

        for step, (data, targets) in enumerate(train_loader):
   
            # prepare
            data = data.to(self.device)
            targets = targets.to(self.device)

            batch_size = data.size(0)

            # training
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item() * batch_size
            N += batch_size

        return loss_sum / N


    def evaluate(self, test_loader):
        self.model.eval()

        loss_sum = 0.0
        N = 0
        outputs_data = list()
        targets_data = list()

        with torch.no_grad():
            for _, (data, targets) in enumerate(test_loader):
    
                data = data.to(self.device)
                targets = targets.to(self.device)
                batch_size = data.size(0)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                loss_sum += loss.item() * batch_size
                N += batch_size

        return loss_sum / N
