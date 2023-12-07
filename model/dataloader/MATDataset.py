from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
import os.path as osp
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
class MARDataset(Dataset):
    def __init__(self, setname, args, augment=False):
        root = osp.join(ROOT_PATH, 'data/MAR/train')
        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
        images_path = Path(root)
       
        images_list = list(images_path.glob('*/*.jpg')) 
        images_list_str = [ str(x) for x in images_list ]

        self.label = []
        for i in os.listdir(root):
            self.label += [i] * len(os.listdir(root + "/" + i))
        self.num_class = len(set(self.label))
        self.label = LabelEncoder().fit_transform(self.label)
        self.data = images_list_str

    def __getitem__(self, item):
        data = self.data[item]

        image = self.transform(Image.open(data).convert('RGB'))

        label = self.label[item]
        return image, label

    def __len__(self):
        return len(self.data)