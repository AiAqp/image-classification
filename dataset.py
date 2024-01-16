import common_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from typing import Callable, Optional, Tuple, Any

class TransformableSubset(Dataset):
    """Dataset wrapper that applies a specified transformation to a subset of a dataset."""
    def __init__(self, subset: Subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def compose_transforms(items):
    transform_list = []
    for item in items:
        type = item.type
        args = item.get('args',{})
        transform_list.append(common_utils.get_object(transforms, type, **args))
    
    return transforms.Compose(transform_list)

def get_datasets(cfg):
    default_transform = compose_transforms(cfg.transforms.default)
    train_transform = compose_transforms(cfg.transforms.training+cfg.transforms.default)

    train_dataset = common_utils.get_object(
        package = datasets, 
        object_name = cfg.dataset.type, 
        root = cfg.dataset.root,
        train = True,
        download = True
    )

    test_dataset = common_utils.get_object(
            package = datasets, 
            object_name = cfg.dataset.type, 
            root = cfg.dataset.root,
            train = False,
            download = True,
            transform = default_transform
        )
    
    validation_size = int(cfg.dataset.validation_split * len(train_dataset))
    train_size = len(train_dataset) - validation_size

    train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size])

    train_dataset = TransformableSubset(train_subset, transform=train_transform)
    validation_dataset = TransformableSubset(validation_subset, transform=default_transform)
    
    return train_dataset, validation_dataset, test_dataset

def get_loaders(train_dataset, validation_dataset, test_dataset, shuffle, n_batches):
    train_loader = DataLoader(train_dataset, batch_size=n_batches, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=n_batches, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=n_batches, shuffle=False)

    return train_loader, validation_loader, test_loader