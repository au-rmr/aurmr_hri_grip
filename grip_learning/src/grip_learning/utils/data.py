import math
import torch
import numpy as np
import cv2

import pytorchvideo
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)



def get_pixel_values(image_processor, image, mask, visual_inputs):
    if visual_inputs == 'rgb':
        image_tensor = torch.FloatTensor(image_processor(image).pixel_values[0])
        return image_tensor
    elif visual_inputs == 'mask':
        mask_tensor = torch.FloatTensor(image_processor(mask).pixel_values[0])
        return mask_tensor
    elif visual_inputs == 'rgb_mask_apply':
        marr = np.array(mask)
        bin_mask = (marr[:,:,2:3] > 0).astype(float)
        image_masked = (bin_mask * np.array(image).astype(float)).astype(np.uint8)
        image_tensor = torch.FloatTensor(image_processor(image_masked).pixel_values[0])
        return image_tensor
    elif visual_inputs == 'rgb_mask_concat':
        image_tensor = torch.FloatTensor(image_processor(image).pixel_values[0])
        mask_tensor = torch.FloatTensor(image_processor(mask).pixel_values[0])
        return torch.cat((image_tensor, mask_tensor), dim=2)
    elif visual_inputs == 'rgb_mask_interp':
        # interpolate image and mask then put them in a tensor
        imarr = np.array(image)
        marr = imarr.copy()
        marr[np.where(np.array(mask)[:,:,2]==0)] = np.array((0.0,0.0,0.0))
        out = cv2.addWeighted(imarr, 0.4, marr, 0.6, 0)
        # out = cv2.addWeighted(np.array(image), 0.55, np.array(mask), 0.45, 0)
        return torch.FloatTensor(image_processor(out).pixel_values[0])
    else:
        raise ValueError(f'Unknown visual inputs: {visual_inputs}')
    
def get_xy_heatmap(x, y, image_size, mask=None, radius=25):
    heatmap = np.zeros((image_size, image_size))
    for u in range(x-radius,x+radius+1):
        for v in range(y-radius,y+radius+1):
            if u < 0 or u >= image_size or v < 0 or v >= image_size:
                continue
            u_distance = abs(x-u)
            v_distance = abs(y-v)
            hyp_distance = math.sqrt(u_distance**2 + v_distance**2)
            intensity = max((radius - hyp_distance), 0.0)
            heatmap[u,v] = (1.0 * intensity) / radius

    if mask is not None:
        marr = np.array(mask)
        bin_mask = (marr[:,:,2:3] > 0).astype(np.float32)
        heatmap = np.expand_dims(heatmap, -1)
        heatmap = (heatmap * bin_mask).astype(np.float32)
        heatmap = heatmap.squeeze(-1)
        
    return heatmap

def get_xy_mask(x, y, image_size, mask=None, radius=10):
    heatmap = np.zeros((image_size, image_size)).astype(np.int64)

    if mask is not None:
        heatmap = (np.array(mask)[:,:,2:3] > 0).squeeze().astype(np.int64)

    for u in range(x-radius,x+radius+1):
        for v in range(y-radius,y+radius+1):
            if u < 0 or u >= image_size or v < 0 or v >= image_size:
                continue
            u_distance = abs(x-u)
            v_distance = abs(y-v)
            hyp_distance = math.sqrt(u_distance**2 + v_distance**2)
            if hyp_distance >= radius:
                continue
            heatmap[u,v] = 2.0

    if mask is not None:
        marr = np.array(mask)
        bin_mask = (marr[:,:,2:3] > 0).astype(np.int64)
        heatmap = np.expand_dims(heatmap, -1)
        heatmap = (heatmap * bin_mask).astype(np.int64)
        heatmap = heatmap.squeeze(-1)

    return heatmap

def preprocess_message_for_classification(bridge, model, image_processor, msg, device='cuda'):
    video_frames = []
    video_frames_cv = []
    for image in msg.images:
        cv_im = bridge.imgmsg_to_cv2(image)
        video_frames.append(torch.FloatTensor(cv_im))
        video_frames_cv.append(cv_im)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 8
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)
    transform = Compose(
        [
            UniformTemporalSubsample(num_frames_to_sample),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            Resize(resize_to),
        ]
    )
    # pytorchvideo.data.make_clip_sampler("uniform", clip_duration)
    inputs = thwc_to_cthw(torch.stack(video_frames)).to(torch.float32)
    inputs = transform(inputs)
    return inputs, video_frames_cv


