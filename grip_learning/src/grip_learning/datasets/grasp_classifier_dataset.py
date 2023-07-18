from typing import Any, Callable, Dict, Optional, Type

from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.clip_sampling import ClipSampler

import torch


GRASP_LABEL_TO_ID = {
    'success': 0,
    'fail_wrong_item': 0,
    'fail_multipick': 1,
    'fail_not_picked': 2
}

ID_TO_GRASP_LABEL = {
    0: 'success',
    1: 'fail_multipick',
    2: 'fail_not_picked'
}


class GraspClassifierDataset(LabeledVideoDataset):
    def __init__(self,
                 data_path: str,
                 clip_sampler: ClipSampler,
                 video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
                 transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 video_path_prefix: str = "",
                 decode_audio: bool = True,
                 decoder: str = "pyav"):

        labeled_video_paths = LabeledVideoPaths.from_path(data_path)
        labeled_video_paths.path_prefix = video_path_prefix
        super().__init__(
            labeled_video_paths,
            clip_sampler,
            video_sampler,
            transform,
            decode_audio=decode_audio,
            decoder=decoder,
        )
