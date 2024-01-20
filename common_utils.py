import torch
import random
import importlib
import numpy as np
from torch.nn import Module
from typing import Any, List, Dict, Optional, Union

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
    

def initialize_from_config(config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Any, List[Any]]:
    """
    Initializes objects based on the provided configuration or a list of configurations.
    Each configuration is dict-like with 'type' and optionally 'args'.
    """
    def initialize_single_object(cfg: Dict[str, Any]) -> Any:
        if 'type' not in cfg:
            raise ValueError("Configuration dictionary must contain a 'type' key.")

        object_type = cfg['type']
        args = cfg.get('args', {})

        object_class = import_object(object_type)
        return object_class(**args) if args else object_class()

    if hasattr(config, 'items'): 
        return initialize_single_object(config)
    elif hasattr(config, '__iter__') and not isinstance(config, str):
        return [initialize_single_object(cfg) for cfg in config]
    else:
        raise TypeError("Config must be a dict-like or list-like object.")