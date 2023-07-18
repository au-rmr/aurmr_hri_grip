import torch
import numpy as np


def get_pixel_values(image_processor, image, mask, visual_features):
    if visual_features == 'rgb':
        image_tensor = torch.FloatTensor(image_processor(image).pixel_values[0])
        return image_tensor
    elif visual_features == 'mask':
        mask_tensor = torch.FloatTensor(image_processor(mask).pixel_values[0])
        return mask_tensor
    elif visual_features == 'rgb_mask_apply':
        marr = np.array(mask)
        bin_mask = (marr[:,:,2:3] > 0).astype(float)
        image_masked = (bin_mask * np.array(image).astype(float)).astype(np.uint8)
        image_tensor = torch.FloatTensor(image_processor(image_masked).pixel_values[0])
        return image_tensor
    elif visual_features == 'rgb_mask_concat':
        image_tensor = torch.FloatTensor(image_processor(image).pixel_values[0])
        mask_tensor = torch.FloatTensor(image_processor(mask).pixel_values[0])
        return torch.cat((mask_tensor, image_tensor), dim=1)
    else:
        raise ValueError(f'Unknown visual features: {visual_features}')