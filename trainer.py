import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter()

    def train(self):
        self.model.train()
        for epoch in range(self.config['epochs']):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Training loop logic
                # ...
                self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.close()