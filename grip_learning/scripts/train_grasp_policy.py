import argparse
import numpy as np
import json
import os

from transformers import TrainingArguments, Trainer, AutoImageProcessor, logging
from grip_learning.datasets.grasp_policy_dataset import GraspPolicyDataset
from grip_learning.models.policy.swinv2 import Swinv2GraspPolicy

def main(args):
    logging.set_verbosity_error()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    default_args = {
        "output_dir": args.output,
        "evaluation_strategy": "steps",
        "num_train_epochs": args.epochs,
        "log_level": "error",
        "report_to": "none",
        "load_best_model_at_end": True,
    }

    model = Swinv2GraspPolicy.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256', num_labels=4, problem_type='regression')
    image_processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')

    train_ds = GraspPolicyDataset(
        '/data/aurmr/grip/datasets/grasp_policy',
        image_processor=image_processor,
        visual_features=args.visual_features)

    eval_ds = GraspPolicyDataset(
        '/data/aurmr/grip/datasets/grasp_policy_eval',
        image_processor=image_processor,
        visual_features=args.visual_features)

    training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    result = trainer.train()
    
    eval_result = model(pixel_values=eval_ds[0]['pixel_values'].unsqueeze(0).cuda())
    eval_label = eval_ds[0]['labels']
    print(eval_result)
    print(eval_label)

    print(f'best checkpoint: {trainer.state.best_model_checkpoint}')

    trainer.save_model(f'{args.output}/best')

    with open(f'{args.output}/config.json', 'w') as f:
        json.dump({
            'image_processor': 'microsoft/swinv2-tiny-patch4-window8-256',
            'visual_features': args.visual_features,
        }, f)

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
    parser.add_argument('-v', '--visual_features', type=str, default='rgb_mask_concat')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    args = parser.parse_args()
    main(args)