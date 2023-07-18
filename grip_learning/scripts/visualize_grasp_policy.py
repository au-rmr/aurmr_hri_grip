import argparse
import numpy as np
import json
import cv2

from transformers import TrainingArguments, Trainer, AutoImageProcessor, logging
from grip_learning.datasets.grasp_policy_dataset import GraspPolicyDataset
from grip_learning.models.policy.swinv2 import Swinv2GraspPolicy

def main(args):
    logging.set_verbosity_error()

    with open(args.config, 'r') as f:
        config = json.load(f)

    model = Swinv2GraspPolicy.from_pretrained(args.model, num_labels=4, problem_type='regression')
    image_processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')

    model.cuda()

    eval_ds = GraspPolicyDataset(
        '/data/aurmr/grip/datasets/grasp_policy_eval',
        image_processor=image_processor,
        visual_features=config['visual_features'])

    for eval_item in eval_ds:
        eval_result = model(pixel_values=eval_item['pixel_values'].unsqueeze(0).cuda())
        eval_label = eval_item['labels']

        image = np.array(eval_item['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(eval_item['mask'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        masked_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

        cv2.circle(masked_image, (int(eval_label[0]), int(eval_label[1])), 5, (0, 0, 255), -1)

        cv2.imshow('image', image)
        cv2.imshow('masked_image', masked_image)
        cv2.waitKey(0)

    print(eval_result)
    print(eval_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Path to model checkpoint')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    args = parser.parse_args()
    main(args)