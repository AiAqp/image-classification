import datasets
import metrics
import common_utils

import mlflow
from tqdm import tqdm
import torch
import warnings
from omegaconf import DictConfig

class ExperimentRunner:
    """
    ExperimentRunner handles the overall experiment pipeline, based on a provided configuration.

    Parameters:
    - cfg (Config): A configuration object containing all settings for the experiment, including dataset details,
                    model configuration, optimizer settings, and training parameters.

    Key Methods:
    - train_step(): Executes a single training step over the training dataset.
    - eval_step(): Evaluates the model on the validation dataset.
    - test_step(): Tests the model on the testing dataset.
    - run(): Runs the training process, including training, validation, and testing phases, and handles early stopping.
    
    The class utilizes `mlflow` for experiment tracking and logging, and `tqdm` for displaying progress during training, 
    validation, and testing.
    """
    def __init__(self, cfg: DictConfig):
        self.run_metrics = {}
        self.cfg = cfg
        self.set_device()
        common_utils.set_random_seeds(cfg.experiment.seed)

        # dataset pipeline
        ds = datasets.get_datasets_from_config(cfg.datasets)
        tr = datasets.get_transforms_from_config(cfg.transforms)
        # load datasets
        self.data = datasets.DataPipeline(*ds)
        # load transforms
        self.data.set_transforms(*tr)
        # load dataset loaders
        self.data.set_loaders(**cfg.dataloaders)

        # initialize model 
        input_shape = self.data.train_dataset[0][0].shape
        n_classes = len(self.data.train_dataset.classes)
        self.model = common_utils.import_object(cfg.model.type)(input_shape, n_classes)

        # initialize optimizer
        self.optimizer = common_utils.import_object(cfg.optimizer.type)(self.model.parameters(), **cfg.optimizer.args)

        # initialize loss tracker 
        criterion = common_utils.initialize_from_config(cfg.criterion)
        callbacks = common_utils.initialize_from_config(cfg.metrics, callback=True)
        self.losses = metrics.LossTracker(criterion, callbacks)

        tqdm.__init__ = (lambda *args, **kwargs: None) if not cfg.experiment.show_progress else tqdm.__init__

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

    def train_step(self) -> None:
        """
        Perform a single training step over the entire training dataset.
        """
        self.losses.reset()
        self.model.train()

        for batch in tqdm(self.data.train_loader, desc="Training"):
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.losses.step(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write(f"Training loss: {self.losses.get_metric('loss')}")

    def eval_step(self) -> None:
        """
        Evaluate the model on the validation dataset.
        """
        self.losses.reset()
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.data.val_loader, desc="Validating"):
                inputs, targets = batch
                outputs = self.model(inputs)
                self.losses.step(outputs, targets)

        tqdm.write(f"Validation loss: {self.losses.get_metric('loss')}")

    def test_step(self) -> None:
        """
        Test the model on the testing dataset.
        """
        self.losses.reset()
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.data.test_loader, desc="Testing"):
                inputs, targets = batch
                outputs = self.model(inputs)
                self.losses.step(outputs, targets)

        tqdm.write(f"Testing loss: {self.losses.get_metric('loss')}")

    def run(self) -> None:
        """
        Execute the complete training, validation, and testing cycle, including handling early stopping.
        """
        name = self.cfg.model.type.rsplit('.', 1)[-1]
        mlflow.set_experiment(experiment_name=name)

        stopping_monitor = metrics.EarlyStoppingMonitor(self.cfg.training.patience)
        name = self.cfg.datasets.type.rsplit('.', 1)[-1]
        with mlflow.start_run(run_name=name):
            mlflow.log_params(self.cfg)
            mlflow.log_params(self.run_metrics)

            for epoch in tqdm(range(self.cfg.training.n_epochs), desc="Epochs"):
                self.train_step()
                mlflow.log_metrics(self.losses.get_metrics('train'), step=epoch)
                

                self.eval_step()
                mlflow.log_metrics(self.losses.get_metrics('val'), step=epoch)

                if not stopping_monitor.update_and_check(self.losses.get_metric('loss')):
                    tqdm.write("Early stopping triggered")
                    break

            self.eval_step()
            mlflow.log_metrics(self.losses.get_metrics('test'), step=epoch)