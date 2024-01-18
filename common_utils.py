import torch
import random
import importlib
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


def import_object(full_path: str, allowed_names: Optional[List[str]] = None) -> Any:
    """
    Dynamically retrieves an object from a given full path (module plus object name).
    """
    module_path, object_name = full_path.rsplit('.', 1)

    if allowed_names is not None and object_name not in allowed_names:
        raise ValueError(f"Object name '{object_name}' is not allowed. Allowed objects are: {allowed_names}")

    try:
        module = importlib.import_module(module_path)

        return getattr(module, object_name)
    
    except AttributeError:
        raise AttributeError(f"Object named '{object_name}' not found in the module '{module_path}'.")
    
    except ImportError:
        raise ImportError(f"Module named '{module_path}' could not be imported.")