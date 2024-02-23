from typing import Callable, Dict, List, Optional, Type
import torch.nn as nn

__all__ = ['list_models', 'get_model']

IMPLEMENTED_MODELS: Dict[str, Type[nn.Module]] = {}

def register_model(name: Optional[str] = None) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a model class in the IMPLEMENTED_MODELS dictionary.
    """
    def wrapper(model: Type[nn.Module]) -> Type[nn.Module]:
        key = name if name is not None else model.__name__
        if key in IMPLEMENTED_MODELS:
            raise ValueError(f"Name '{key}' already registered.")
        IMPLEMENTED_MODELS[key] = model
        return model

    return wrapper

def list_models() -> List[str]:
    """
    Returns a list of currently implemented models.
    """
    return list(IMPLEMENTED_MODELS.keys())

def get_model(name:str, **kwargs) -> nn.Module:
    """
    Instantiate and return a model by name, with optional keyword arguments.
    """
    try:
        model = IMPLEMENTED_MODELS[name]
    except KeyError:
        raise ValueError(f"Model {name} not implemented")
    
    return model(**kwargs)