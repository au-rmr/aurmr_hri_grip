
from typing import Optional, Tuple, Union
import os
import json

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import Swinv2PreTrainedModel, Swinv2Model, AutoImageProcessor


class Swinv2GraspPolicy(Swinv2PreTrainedModel):

    @staticmethod
    def from_pretrained(path, return_processor_and_config=False, **kwargs):
        model = super(Swinv2GraspPolicy, Swinv2GraspPolicy).from_pretrained(path, **kwargs)

        if not return_processor_and_config:
            return model
        
        with open(os.path.join(path, '../config.json'), 'r') as f:
            config = json.load(f)

        image_processor = AutoImageProcessor.from_pretrained(config['image_processor'])

        return model, image_processor, config

    def __init__(self, config):
        super().__init__(config)

        self.swinv2 = Swinv2Model(config, add_pooling_layer=False, use_mask_token=True)

        self.num_labels = config.num_labels
        self.swinv2 = Swinv2Model(config)

        # # Classifier head
        # self.classifier = (
        #     nn.Linear(self.swinv2.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # )

        # Grip (x,y,z,w) regression head
        self.grip_fc1 = nn.Linear(self.swinv2.num_features, 256)
        self.grip_fc2 = nn.Linear(256, 4)

        # Grip classification head
        # TODO

        # Grip segmentation head
        # TODO

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        target_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swinv2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        # logits = self.classifier(pooled_output)
        logits = self.grip_fc1(pooled_output)
        logits = self.grip_fc2(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'reshaped_hidden_states': outputs.reshaped_hidden_states,
        }