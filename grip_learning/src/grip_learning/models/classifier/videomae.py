from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import VideoMAEPreTrainedModel, VideoMAEModel
from transformers.modeling_outputs import ImageClassifierOutput

class VideoMAEForGraspClassification(VideoMAEPreTrainedModel):
    def __init__(self, config, use_deepspeed=False):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.videomae = VideoMAEModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Deepspeed
        if use_deepspeed:
            self.videomae.half()
            if self.fc_norm:
                self.fc_norm.half()
            self.classifier.half()
        self.use_deepspeed = use_deepspeed

        # Initialize weights and apply final processing
        self.post_init()

    # def save_checkpoint(self, output_dir: str):
    #     self.save_pretrained(output_dir)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoImageProcessor, VideoMAEForVideoClassification
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 16 frames
        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        >>> model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        eating spaghetti
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.use_deepspeed:
            pixel_values = pixel_values.half()

        outputs = self.videomae(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                self.config.problem_type = "single_label_classification"

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )