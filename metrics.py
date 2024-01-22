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
            self.metrics[callback.__name__] += callback(predictions, targets)

        self._counter += 1

        return loss
        
    def get_metrics(self, prefix: Optional[str] = None) -> Dict[str, float]:
        """Returns the calculated metrics, optionally prefixed."""
        divider = 1 if self._counter == 0 else self._counter
        return {prefix+'_'+key if prefix else key: value/divider for key,value in self.metrics.items()}
    
    def reset(self) -> None:
        """Resets all tracked metrics to 0."""
        self._counter = 0
        for key in self.metrics:
            self.metrics[key] = 0.0
