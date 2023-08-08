import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import json
import numpy as np

from grip_learning.utils.data import get_pixel_values, get_xy_heatmap, get_xy_mask

WIDTH_TO_CLASS_ID = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
    1.5: 3,
    2.0: 4,
    2.5: 5,
    3.0: 6,
    3.5: 7,
    4.0: 8,
    4.5: 9,
    5.0: 10,
}

Z_PADDING_TO_CLASS_ID = {
    0.0: 0,
    5.0: 1,
    10.0: 2,
    -5.0: 3,
    -10.0: 4,
}

def round_off_rating(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return 

class GraspPolicyDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_processor=None, \
                 visual_inputs='rgb_mask_concat', label_type='continuous'):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_processor = image_processor
        self.metadata_paths = glob.glob(dataset_path + '/*.json')
        self.visual_inputs = visual_inputs

        assert label_type in ['continuous', 'discrete_xy', 'discrete_z', 'discrete_w']
        self.label_type = label_type

    def __len__(self):
        return len(self.metadata_paths)

    def __getitem__(self, idx):
        metadata_path = self.metadata_paths[idx]
        image_path = metadata_path.replace('.json', '.png')
        mask_path = metadata_path.replace('.json', '_mask.png')
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'width' not in metadata:
            metadata['width'] = 0.015

        x = round(metadata['x'] * image.size[0])
        y = round(metadata['y'] * image.size[1])
        z_padding = metadata['z_padding'] * 100
        width = metadata['width'] * 100

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.FloatTensor([x,y,z_padding,width])

        pixel_values = get_pixel_values(self.image_processor, image, mask, self.visual_inputs)

        if self.label_type == 'discrete_xy':
            labels = get_xy_mask(x, y, image.size[0], mask=mask, radius=10)
        
        if self.label_type == 'discrete_z':
            labels = [Z_PADDING_TO_CLASS_ID[float(z_padding)]]
        
        if self.label_type == 'discrete_w':
            labels = [WIDTH_TO_CLASS_ID[float(round(width * 2) / 2)]]

        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'image': image,
            'mask': mask
        }