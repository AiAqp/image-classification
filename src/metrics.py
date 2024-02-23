from typing import List, Callable, Dict, Optional
import torch.nn as nn

class LossTracker:
    """Tracks and calculates loss and other metrics during model training."""
    def __init__(self, criterion: Callable, metric_callbacks: List[Callable]) -> None:
        self._counter = 0
        self.metrics = {}
        self.criterion = criterion
        self.callbacks = metric_callbacks

        self.metrics['loss'] = 0.0
        for callback in metric_callbacks:
            self.metrics[callback.__name__] = 0.0
    
    def step(self, predictions, targets) -> nn.Module:
        """Updates metrics and returns loss, based on the provided predictions and targets."""
        loss = self.criterion(predictions, targets)

        self.metrics['loss'] += loss.item()
        for callback in self.callbacks:
            self.metrics[callback.__name__] += callback(targets, predictions.argmax(dim=-1))

        self._counter += 1

        return loss
    
    def get_metric(self, name:str) -> float:
        return self.metrics[name] / self._counter
        
    def get_metrics(self, prefix: Optional[str] = None) -> Dict[str, float]:
        """Returns the calculated metrics, optionally prefixed."""
        divider = 1 if self._counter == 0 else self._counter
        return {prefix+'_'+key if prefix else key: value / divider for key, value in self.metrics.items()}
    
    def summary(self) -> str:
        pass
    
    def reset(self) -> None:
        """Resets all tracked metrics to 0."""
        self._counter = 0
        for key in self.metrics:
            self.metrics[key] = 0.0


class EarlyStoppingMonitor:
    """
    Monitors a single metric and checks for early stopping condition.
    """
    def __init__(self, patience: int) -> None:
        self.min_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

    def update_and_check(self, loss: float) -> bool:
        """Updates the monitor with the current loss and checks for early stopping condition.

        Args:
            loss: The current loss value to be checked.

        Returns:
            bool: True if training should continue, False if the early stopping condition is met.
        """
        if loss < self.min_loss:
            self.min_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
        return True