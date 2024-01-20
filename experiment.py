import datasets
import models
import common_utils

import torch
import warnings

class ExperimentRunner:
    def __init__(self, cfg):
        self.run_metrics = {}
        self.cfg = cfg
        self.set_device()

        # initialize dataset pipeline
        ds = datasets.get_datasets_from_config(cfg.datasets)
        tr = datasets.get_transforms_from_config(cfg.transforms)
        self.data = datasets.DataPipeline(*ds)
        self.data.set_transforms(*tr)
        self.data.set_loaders(**cfg.dataloaders)

        # initialize model 
        input_shape = self.data.train_dataset[0][0].shape
        n_classes = len(self.data.train_dataset.classes)
        self.model = common_utils.import_object(cfg.model.type)(input_shape, n_classes)

        # initialize optimizer
        self.optimizer = common_utils.import_object(cfg.optimizer.type)(self.model.parameters(), **cfg.optimizer.args)

        # initialize loss 
        self.criterion = common_utils.initialize_from_config(cfg.criterion)

        common_utils.set_random_seeds(cfg.experiment.seed)

    def set_device(self) -> None:
        """
        Sets the device for the experiment based on configuration and system availability.
        Appends the device information to run_metrics and issues a warning if GPU is requested but not available.
        """
        if self.cfg.experiment.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if self.cfg.experiment.use_gpu:
                warnings.warn("GPU requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.run_metrics["device"] = str(self.device)

    def run(self):
        pass

def train_step(model, dataloader, objective, optimizer):
    pass

def eval_step(model, dataloader, objective, optimizer):
    pass