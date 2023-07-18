import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import json
import numpy as np

from grip_learning.utils.data import get_pixel_values

class GraspPolicyDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_processor=None, visual_features='rgb_mask_concat'):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_processor = image_processor
        self.metadata_paths = glob.glob(dataset_path + '/*.json')
        self.visual_features = visual_features

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

        pixel_values = get_pixel_values(self.image_processor, image, mask, self.visual_features)

        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'image': image,
            'mask': mask
        }