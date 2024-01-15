import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# A dictionary registering the available datasets along with their number of classes.
_dataset_register = {
    'CIFAR-10': (datasets.CIFAR10, 10),
    'CIFAR-100': (datasets.CIFAR100, 100)
}

def load_dataset(dataset_config):
    dataset_name = dataset_config['dataset']
    if dataset_name not in _dataset_register:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")

    loader, n_classes = _dataset_register[dataset_name]

    train_loader, test_loader = _create_loaders(loader, dataset_config)
    return train_loader, test_loader, n_classes

def _create_loaders(loader, dataset_config):
    train_transform, test_transform = _get_transforms()

    train_set = loader(dataset_config['root'], train=True, download=True, transform=train_transform)
    test_set = loader(dataset_config['root'], train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, dataset_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, dataset_config['batch_size'], shuffle=False)

    return train_loader, test_loader

def _get_transforms(custom_transforms=None):
    if custom_transforms:
        return custom_transforms

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, test_transform
