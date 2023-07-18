import argparse
import numpy as np
import json
import os
import torch
import evaluate
from functools import partial

import pytorchvideo
import pytorchvideo.data

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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers import TrainingArguments, Trainer, AutoImageProcessor, logging, VideoMAEImageProcessor, VideoMAEForVideoClassification
from grip_learning.datasets.grasp_classifier_dataset import GraspClassifierDataset, GRASP_LABEL_TO_ID, ID_TO_GRASP_LABEL
from grip_learning.models.classifier.videomae import VideoMAEForGraspClassification
from grip_learning.utils.optim import get_parameter_groups, LayerDecayValueAssigner

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def main(args):
    logging.set_verbosity_error()

    default_args = {
        "output_dir": args.output,
        "evaluation_strategy": "steps",
        "num_train_epochs": 30,
        "log_level": "error",
        "report_to": "none",
        "load_best_model_at_end": True,
    }

    model_ckpt = "MCG-NJU/videomae-base-ssv2"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForGraspClassification.from_pretrained(
        model_ckpt,
        label2id=GRASP_LABEL_TO_ID,
        id2label=ID_TO_GRASP_LABEL,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        use_deepspeed=args.deepspeed_config is not None
    )

    num_frames_to_sample = model.config.num_frames
    sample_rate = 8
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    # if args.deepspeed_config is not None:
    #     model_config = model.config
    #     import deepspeed
    #     layer_decay = 0.7 # TODO move to config or args
    #     weight_decay = 0.05 # TODO move to config or args
    #     distributed = False
    #     skip_weight_decay_list = ['cls_token', 'pos_embed']

    #     num_layers = model.config.num_hidden_layers
    #     assigner = LayerDecayValueAssigner(list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    #     # loss_scaler = None
    #     optimizer_params = get_parameter_groups(
    #         model, weight_decay, skip_weight_decay_list,
    #         assigner.get_layer_id if assigner is not None else None,
    #         assigner.get_scale if assigner is not None else None)

    #     model, optimizer, _, _ = deepspeed.initialize(
    #         args=args, model=model, model_parameters=optimizer_params, dist_init_required=not distributed,
    #     )

    #     # TODO figure out how to get these from the checkpoint
    #     num_warmup_steps=152
    #     num_training_steps=1520

    #     lr_lambda = partial(
    #         _get_linear_schedule_with_warmup_lr_lambda,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )
    #     #import pdb; pdb.set_trace()
    #     last_epoch = -1 # TODO: figure out how to get this from the checkpoint
    #     lr_scheduler = LambdaLR(optimizer.optimizer, lr_lambda, last_epoch)
        
    #     model.config = model_config

    #     def clip_grad_norm_(max_norm):
    #         return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    #     optimizer.clip_grad_norm = clip_grad_norm_

    # TODO: recalc stats for our data (?)
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    train_ds = GraspClassifierDataset(
        data_path=os.path.join(args.input, 'train'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    eval_ds = GraspClassifierDataset(
        data_path=os.path.join(args.input, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_ds = GraspClassifierDataset(
        data_path=os.path.join(args.input, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    num_epochs = 20
    batch_size = 8 if args.deepspeed_config is None else 4

    training_args = TrainingArguments(
        args.name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        max_steps=(train_ds.num_videos // batch_size) * num_epochs,
        deepspeed={
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True
            },
            # "train_batch_size": 10,
            "train_micro_batch_size_per_gpu": batch_size,
            "steps_per_print": 1000,
            # "optimizer": {
            #     "type": "Adam",
            #     "adam_w_mode": True,
            #     # "params": {
            #     #     "lr": 0.001,
            #     #     "weight_decay": 0.05,
            #     #     "bias_correction": True,
            #     #     "betas": [
            #     #         0.9,
            #     #         0.999
            #     #     ],
            #     #     "eps": 1e-08
            #     # }
            # },
            # "fp16": {
            #     "enabled": True,
            #     "loss_scale": 0,
            #     "initial_scale_power": 7,
            #     "loss_scale_window": 128
            # }
        } if args.deepspeed_config is not None else None,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        # optimizers=(optimizer, lr_scheduler) if args.deepspeed_config is not None else (None, None)
    )
    result = trainer.train()
    
    # eval_sample = next(iter(eval_ds))
    # eval_result = model(pixel_values=eval_sample['pixel_values'].unsqueeze(0).cuda())
    # eval_label = eval_sample['labels']
    # print(eval_result)
    # print(eval_label)

    print(f'best checkpoint: {trainer.state.best_model_checkpoint}')

    trainer.save_model(f'{args.output}/best')

    # with open(f'{args.output}/config.json', 'w') as f:
    #     json.dump({
    #         'image_processor': 'microsoft/swinv2-tiny-patch4-window8-256',
    #         'visual_features': args.visual_features,
    #     }, f)

    # import pdb; pdb.set_trace()


    # TODO: figure out how to evaluate the model (just use a train set item for now)
    # TODO: train for multiple epochs and test model
    # TODO: save best model
    # TODO: add evaluation (pull out some of the images from the training set and evaluate on them)
    # TODO: allow use of other encoders (e.g. ViT, ResNet, Mask2Former)
    # TODO: allow use of multimodal encoders (e.g. CLIP, FLAVA)
    # print_summary(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='Path to output directory', default='/tmp')
    parser.add_argument('-i', '--input', type=str, help='Path to input directory', default='/tmp')
    parser.add_argument('-n', '--name', type=str, help='Model name', default='grasp_classifier')
    parser.add_argument('--deepspeed_config', default=None, type=str, help='DeepSpeed json configuration file.')
    # parser.add_argument('-v', '--visual_features', type=str, default='rgb_mask_concat')
    args = parser.parse_args()
    main(args)