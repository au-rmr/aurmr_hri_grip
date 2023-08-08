import argparse
import numpy as np
import json
import os

from transformers import TrainingArguments, Trainer, AutoImageProcessor, logging
from grip_learning.datasets.grasp_policy_dataset import GraspPolicyDataset
# from grip_learning.models.policy.swinv2 import Swinv2GraspPolicy
from grip_learning.models.policy_model import GraspPolicyModel

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

    num_labels = 4
    if args.model_output == 'discrete_xy':
        num_labels = 1

    model = GraspPolicyModel.from_pretrained(
        args.pretrained_features,
        num_labels=num_labels,
        problem_type='regression',
        image_size=args.image_size,
        visual_inputs=args.visual_inputs,
        output_type=args.model_output,
        return_dict=True)

    image_processor = AutoImageProcessor.from_pretrained(
        args.pretrained_features,
        image_size=args.image_size)

    # image_processor.size['width'] = args.image_size
    # image_processor.size['height'] = args.image_size

    train_ds = GraspPolicyDataset(
        args.training_data,
        image_processor=image_processor,
        visual_inputs=args.visual_inputs,
        label_type=args.model_output)

    eval_ds = GraspPolicyDataset(
        args.validation_data,
        image_processor=image_processor,
        visual_inputs=args.visual_inputs,
        label_type=args.model_output)

    training_args = TrainingArguments(per_device_train_batch_size=args.batch_size, **default_args)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    result = trainer.train()

    import pdb; pdb.set_trace()
    
    eval_result = model(pixel_values=eval_ds[0]['pixel_values'].unsqueeze(0).cuda())
    eval_label = eval_ds[0]['labels']
    print(eval_result)
    print(eval_label)

    print(f'best checkpoint: {trainer.state.best_model_checkpoint}')

    trainer.save_model(f'{args.output}/best')

    with open(f'{args.output}/config.json', 'w') as f:
        json.dump({
            'image_processor': 'microsoft/swinv2-tiny-patch4-window8-256',
            'visual_inputs': args.visual_inputs,
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
    parser.add_argument('-t', '--training_data', type=str, help='Path to training data directory', default='/data/aurmr/grip/datasets/grasp_policy')
    parser.add_argument('-v', '--validation_data', type=str, help='Path to eval data directory', default='/data/aurmr/grip/datasets/grasp_policy_eval')
    parser.add_argument('-o', '--output', type=str, help='Path to output directory', default='/tmp')
    parser.add_argument('-m', '--model_type', type=str, default='swinv2')
    parser.add_argument('-pt', '--pretrained_features', type=str, default='microsoft/swinv2-tiny-patch4-window8-256')
    parser.add_argument('-vi', '--visual_inputs', type=str, default='rgb_mask_concat')
    parser.add_argument('-mo', '--model_output', type=str, default='continuous')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=252)
    args = parser.parse_args()
    main(args)