import torch
import random
import numpy as np
from torch.nn import Module
from typing import Any, List, Dict, Optional, TypeVar

def set_random_seeds(seed: int) -> None:
    """
    Set the random seeds for reproducibility in numpy, random, and PyTorch.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model: Module, path: str) -> None:
    """
    Save the state dictionary of a PyTorch model to a specified file path.
    """
    torch.save(model.state_dict(), path)


ModuleType = TypeVar('ModuleType')

def get_object(object_name: str, package: ModuleType, allowed_names: Optional[List[str]] = None, **kwargs) -> Any:
    """
    Dynamically retrieves an object from a given package and initializes it with provided keyword arguments.
    """
    if allowed_names is not None and object_name not in allowed_names:
        raise ValueError(f"Object name '{object_name}' is not allowed. Allowed objects are: {allowed_names}")

    try:
        obj = getattr(package, object_name)
        return obj(**kwargs)
    except AttributeError:
        raise AttributeError(f"Object named '{object_name}' not found in the provided package.")
