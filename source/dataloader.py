from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import TEST_DIR, TRAIN_DIR, VAL_DIR, BATCH_SIZE, IMG_SIZE

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return train_transforms, test_transforms

def get_dataloaders():
    train_transforms, test_transfors = get_transforms()
    
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_data = datasets.ImageFolder(VAL_DIR, transform=test_transfors)
    test_data = datasets.ImageFolder(TEST_DIR, transform=test_transfors)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader