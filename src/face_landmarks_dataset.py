import os

import torch
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import skimage

from torch.utils.data import Dataset
from PIL import Image
from skimage import io, transform

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset. Contains custom transforms to ensure same transforms for image and landmarks"""

    def __init__(self, csv_file, root_dir, augment_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([Rescale(40), RandomCrop(32)])
        self.normalize_to_tensor = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.4467, 0.3882, 0.3499), std=(0.2561, 0.2378, 0.2260))])
        self.augment_data = augment_data

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read image.
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
       
        # Read labels.
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        original_size = landmarks.size
        # Reshape for easier augmenation.
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        image_id = 'image' 
        target_id = 'landmarks'
        sample = {image_id: image, target_id: landmarks}

        # Transform both input (images) and targets (landmarks)
        if self.augment_data:
            sample = self.transform(sample)

        image_normalized = self.normalize_to_tensor(skimage.img_as_float32(sample[image_id]))
        
        # Restore original shape.
        landmarks = sample[target_id]
        landmarks = landmarks.astype('float').reshape(original_size)
        targets = torch.from_numpy(landmarks).float()
        
        return image_normalized, targets

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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}