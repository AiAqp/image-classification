import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_transforms():
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

def load_cifar10(root, batch_size, train_transform, test_transform):
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_data_loaders(config):
    train_transform, test_transform = get_transforms()
    root = config['data_dir']
    batch_size = config['batch_size']

    train_loader, test_loader = load_cifar10(root, batch_size, train_transform, test_transform)
    
    return train_loader, test_loader