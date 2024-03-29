from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import csv

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, root_dir, mode='train', suffix='', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = pd.read_csv(os.path.join(root_dir, f'{mode}_image{suffix}.csv'), sep=',')
        self.root_dir = root_dir
        self.transform = transform
        self.similarity = np.load(os.path.join(root_dir, f'{mode}_similarity{suffix}.npy'))
        self.clip_image_features = np.load(os.path.join(root_dir, f'{mode}_clip_image_features{suffix}.npy'))
        self.clip_text_features = np.load(os.path.join(root_dir, f'{mode}_clip_text_features{suffix}.npy'))

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        img_name = str(os.path.join(self.root_dir, "images",str(self.images_frame.iloc[idx, 0])))
        im = Image.open(img_name).convert('RGB')
        if im.mode == 'P':
            im = im.convert('RGB')
            
        sample = {
            'image': np.asarray(im),
            'rating': self.images_frame.iloc[idx, 1],
            'similarity': self.similarity[idx],
            'clip_image_features': self.clip_image_features[idx],
            'clip_text_features': self.clip_text_features[idx],
        }

        if self.transform:
            sample = self.transform(sample)
        return sample




class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return dict(sample, image=image)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating, similarity = sample['image'], sample['rating'], sample['similarity']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return dict(sample, image=image)


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating, similarity = sample['image'], sample['rating'], sample['similarity']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return dict(sample, image=image)


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating, similarity = sample['image'], sample['rating'], sample['similarity']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return dict(sample, image=image)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating, similarity = sample['image'], sample['rating'], sample['similarity']

        remaining = { key: torch.tensor(sample[key]) for key in sample if key not in ['image', 'rating', 'similarity'] }

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        image, rating = torch.from_numpy(image).double(), torch.from_numpy(np.float64([rating])).double()
        similarity = torch.unsqueeze(torch.tensor(similarity), -1)
        return { 'image': image, 'rating': rating, 'similarity': similarity, **remaining }

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
    
def AI1k_load_data(mode='train', split=0):
    data_dir = os.path.join('AI-1k/')

    output_size = (224, 224)

    if mode == 'train':
        train_path = os.path.join(data_dir, f'train_image_split_{split}.csv')
        transformed_dataset_train = ImageRatingsDataset(root_dir=data_dir,
                                                        mode=mode,
                                                        suffix=f'_split_{split}',
                                                        transform=transforms.Compose([
                                                            Rescale(output_size=(256, 256)),
                                                            RandomHorizontalFlip(0.5),
                                                            RandomCrop(output_size=output_size),
                                                            Normalize(),
                                                            ToTensor(),
                                                            ]))
        dataloader = DataLoader(transformed_dataset_train, batch_size=50,
                                shuffle=False, num_workers=0, collate_fn=my_collate)
    elif mode == 'val':
        val_path = os.path.join(data_dir, f'val_image_split_{split}.csv')
        transformed_dataset_valid = ImageRatingsDataset(root_dir=data_dir,
                                                        mode=mode,
                                                        suffix=f'_split_{split}',
                                                        transform=transforms.Compose([
                                                            Rescale(output_size=(224, 224)),
                                                            Normalize(),
                                                            ToTensor(),
                                                            ]))
        dataloader = DataLoader(transformed_dataset_valid, batch_size=50,
                                shuffle=False, num_workers=0, collate_fn=my_collate)
    elif mode == 'test':  # 'test' mode
        test_path = os.path.join(data_dir, 'test_image.csv')
        transformed_dataset_test = ImageRatingsDataset( root_dir=data_dir,
                                                        mode=mode,
                                                        transform=transforms.Compose([
                                                            Rescale(output_size=(224, 224)),
                                                            Normalize(),
                                                            ToTensor(),
                                                            ]))
        dataloader = DataLoader(transformed_dataset_test, batch_size=50,
                                shuffle=False, num_workers=0, collate_fn=my_collate)
    elif mode == 'all':

        all_path = os.path.join(data_dir, 'all_image.csv')
        transformed_dataset_all = ImageRatingsDataset( root_dir=data_dir,
                                                        mode=mode,
                                                        transform=transforms.Compose([
                                                            Rescale(output_size=(224, 224)),
                                                            Normalize(),
                                                            ToTensor(),
                                                            ]))
        dataloader = DataLoader(transformed_dataset_all, batch_size=50,
                                shuffle=False, num_workers=0, collate_fn=my_collate)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return dataloader
