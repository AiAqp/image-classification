from dataset import get_data_loaders
from model import BasicCNN
from trainer import Trainer
from tester import Tester
from utils import set_random_seeds, save_model

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        set_random_seeds(self.config['seed'])
        self.data_loaders = get_data_loaders(self.config['data'])
        self.model = BasicCNN()
        self.trainer = Trainer(self.model, self.data_loaders['train'], self.config['training'])
        self.tester = Tester(self.model, self.data_loaders['test'], self.config['testing'])

    def save_metrics():
        pass

    def load_model():
        pass

    def run(self):
        self.trainer.train()
        test_metrics = self.tester.test()
        save_model(self.model, self.config['model_path'])
