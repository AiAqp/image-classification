import common_utils
from torchvision.transforms import Compose
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from typing import Callable, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class DataPipeline:
    """ Manages data pipelines for training, validation, and testing datasets. """
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    
    train_loader: DataLoader = field(default=None, init=False)
    val_loader: DataLoader = field(default=None, init=False)
    test_loader: DataLoader = field(default=None, init=False)

    @property
    def datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader, self.val_loader, self.test_loader

    @property
    def default_transform(self) -> Callable:
        return self.test_dataset.transform
    
    @property
    def train_transform(self) -> Callable:
        return self.train_dataset.transform

    @default_transform.setter
    def default_transform(self, transform: Callable):
        """ Sets the default transform for validation and test datasets. """
        self.val_dataset.transform = transform
        self.test_dataset.transform = transform

    @train_transform.setter
    def train_transform(self, transform: Callable):
        self.train_dataset.transform = transform

    def set_transforms(self, default_transform: Callable, train_transform: Optional[Callable] = None):
        """ Sets transformations for all datasets, with an optional specific transform for training. """
        self.default_transform = default_transform
        if train_transform:
            self.train_transform = train_transform
        else:
            self.train_transform = default_transform

    def set_loaders(self, n_batches: int, shuffle: bool):
        """ Initializes DataLoaders for all datasets with specified batch size and shuffle flag. """
        self.train_loader = DataLoader(self.train_dataset, batch_size=n_batches, shuffle=shuffle)
        self.val_loader = DataLoader(self.val_dataset, batch_size=n_batches, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=n_batches, shuffle=False)


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
    
    def __getattr__(self, attr):
        # Proxy any missing attribute or method calls to the original dataset
        return getattr(self.subset.dataset, attr)
    
    @property
    def data(self):
        return [self.subset[i][0] for i in range(len(self.subset))]
    
    @property
    def targets(self):
        return [self.subset[i][1] for i in range(len(self.subset))]


def split_dataset(dataset:Dataset, split:float) -> Tuple[Dataset, Dataset]:
    size1 = int(split * len(dataset))
    size2 = len(dataset) - size1

    subsets = random_split(dataset, [size1, size2])
    dataset1 = TransformableSubset(subsets[0], transform=dataset.transform)
    dataset2 = TransformableSubset(subsets[1])

    return dataset1, dataset2


def get_transforms_from_config(config: dict) -> Tuple[Compose, Optional[Compose]]:
    """
    Creates and returns default and optionally augmented transform pipelines based on the provided configuration.
    """
    default_config = config['default']
    augment_config = config.get('augment')

    default_transforms = common_utils.initialize_from_config(default_config)
    default_compose = Compose(default_transforms)

    if augment_config:
        augment_transforms = common_utils.initialize_from_config(default_config+augment_config)
        augment_compose = Compose(augment_transforms)
    else:
        augment_compose = None
     
    return default_compose, augment_compose


def get_datasets_from_config(cfg):
    dataset = common_utils.import_object(cfg.type)

    train_dataset = dataset(root=cfg.root, train=True, download=True)
    train_dataset, val_dataset = split_dataset(train_dataset, cfg.validation_split)

    test_dataset = dataset(root=cfg.root, train=False, download=True)

    return train_dataset, val_dataset, test_dataset