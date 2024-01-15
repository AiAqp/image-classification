from dataset import load_dataset
import models
from trainer import Trainer
from tester import Tester
from common_utils import set_random_seeds, save_model

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        set_random_seeds(self.config['seed'])
        self.data_loaders, n_classes = load_dataset(self.config['data'])
        self.model = models.get_model('BasicCNN', n_classes=n_classes)
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
