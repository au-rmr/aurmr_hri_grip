
from typing import Optional, Tuple, Union
import os
import json
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    Swinv2PreTrainedModel, Swinv2Model, Swinv2Config,
    Dinov2PreTrainedModel, Dinov2Model, Dinov2Config,
    Dinov2ForImageClassification,
    Swinv2ForImageClassification,
    UperNetPreTrainedModel,
    AutoBackbone,
    AutoImageProcessor
)

from transformers.models.upernet.modeling_upernet import UperNetHead, UperNetFCNHead
from transformers.modeling_outputs import SemanticSegmenterOutput

from grip_learning.datasets.grasp_policy_dataset import WIDTH_TO_CLASS_ID, Z_PADDING_TO_CLASS_ID


class SegmentationGraspModel(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = AutoBackbone.from_config(config.backbone_config)

        # Segmentation head(s)
        self.decode_head = UperNetHead(config, in_channels=self.backbone.channels)
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        features = outputs.feature_maps

        logits = self.decode_head(features)
        logits = nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
            )

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # compute weighted loss
                loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
                main_loss = loss_fct(logits, labels)
                auxiliary_loss = loss_fct(auxiliary_logits, labels)
                loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GraspModel:
    @staticmethod
    def from_pretrained(path, output_type='continuous', model_type=None, return_processor=False, **kwargs):
        if model_type is None:
            if 'swinv2' in path:
                model_type = 'swinv2'
            if 'dinov2' in path:
                model_type = 'dinov2'
        
        kwargs['ignore_mismatched_sizes'] = True
        
        if output_type == 'continuous':
            kwargs['problem_type'] = 'regression'
            kwargs['num_labels'] = 4
        elif output_type == 'discrete_z':
            kwargs['problem_type'] = 'single_label_classification'
            kwargs['num_labels'] = len(Z_PADDING_TO_CLASS_ID.keys())
        elif output_type == 'discrete_w':
            kwargs['problem_type'] = 'single_label_classification'
            kwargs['num_labels'] = len(WIDTH_TO_CLASS_ID.keys())
        elif output_type == 'discrete_xy':
            raise ValueError('discrete_xy not supported, use SegmentationGraspModel instead')
        else:
            raise ValueError(f'Invalid output type {output_type}')

        if model_type == 'swinv2':
            model = Swinv2ForImageClassification.from_pretrained(path, **kwargs)
        elif model_type == 'dinov2':
            model = Dinov2ForImageClassification.from_pretrained(path, **kwargs)
        else:
            raise ValueError(f'Invalid model type {model_type}')
    
        
        
        if return_processor:
            return model, AutoImageProcessor.from_pretrained(model.config._name_or_path)

        return model

