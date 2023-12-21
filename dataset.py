import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

_dataset_loaders = {
    'CIFAR-10' : datasets.CIFAR10,
    'CIFAR-100' : datasets.CIFAR100
}

def load_dataset(dateset_config):
    """Load datasets based on the provided configuration."""
    loader = _dataset_loaders[dateset_config['dataset']]
    train_transform, test_transform = _get_transforms()

    train_set = loader(dateset_config['root'], train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, dateset_config['batch_size'], shuffle=True)

    test_set = loader(dateset_config['root'], train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, dateset_config['batch_size'], shuffle=False)

    return train_loader, test_loader

def _get_transforms(custom_transforms = None):
    """Get default or custom transforms for training and testing."""
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