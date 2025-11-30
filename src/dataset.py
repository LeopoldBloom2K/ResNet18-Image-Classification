import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    학습 및 검증용 데이터로더를 생성합니다.
    """
    # 전처리 정의 (노트북과 동일)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
    }

    image_datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names