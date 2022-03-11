import torch
import torchvision
from torchvision import models, transforms

transform_dict = {
    'grey': transforms.Compose(
        [transforms.Resize((600, 600)),
         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Normalize([0.485], [0.5])
         ]),
    'color': transforms.Compose(
        [transforms.Resize((704, 704)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])}


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def load_images(_dataset_path, color_or_grey: str, _batch_size=60, _num_workers=1):
    _dataset = ImageFolderWithPaths(_dataset_path, transform_dict[color_or_grey])
    _dataloader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers)
    _dataset = ImageFolderWithPaths(_dataset_path, transform_dict[color_or_grey])
    _dataloader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers)

    return _dataset, _dataloader


def create_dataset(train_dir: str, val_dir: str, color_or_grey: str, batch_size: int, num_workers: int):
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform_dict[color_or_grey])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform_dict[color_or_grey])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # log_dir = os.path.join('/content/drive/MyDrive/Institute Phy/expriment_withiut_14.07/log/', log_folder)
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # writer = SummaryWriter(log_dir)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, val_dataloader
