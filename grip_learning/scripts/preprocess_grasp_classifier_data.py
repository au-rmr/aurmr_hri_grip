import rospy
import rosbag
import json
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import argparse


bridge = CvBridge()


CLASS_STRING_TO_INT = {
    'success': 0,
    'fail_wrong_item': 0,
    'fail_multipick': 1,
    'fail_not_picked': 2
}

CLASS_STRING_TO_INT_BINARY = {
    'success': 0,
    'fail_wrong_item': 0,
    'fail_multipick': 1,
    'fail_not_picked': 1
}


def metadata_to_dict(metadata):
    md_dict = {}
    for md in metadata:
        md_dict[md.key] = md.value
    return md_dict


def process_pick_folder(pick_folder, output_folder):

    if not os.path.exists(os.path.join(pick_folder, 'pick_events.json')):
        print(f'No pick events JSON found in {pick_folder}')
        return []

    with open(os.path.join(pick_folder, 'pick_events.json'), 'r') as f:
        pick_events = json.load(f)

    pick_uuid = '_'.join(pick_folder.split('/')[-2:])
    
    # probe_events = [e for e in pick_events['events'] if 'primitive_name' in e['metadata'] and e['metadata']['primitive_name'] == 'probe_super']
    # pick_eval_events = [e for e in pick_events['events'] if e['event_type'] == 'pick_eval' and e['metadata']['ignore'] != 'true']

    # probe_uuids = [e['metadata']['primitive_uuid'] for e in probe_events]
    # eval_outcomes = [e['metadata']['eval_code'] for e in pick_eval_events]
    # eval_item_ids = [e['metadata']['item_id'] for e in pick_eval_events]
    # probe_topics = [f'/probe_super/{uuid}' for uuid in probe_uuids]
    # print(len(probe_uuids))
    # print(len(eval_outcomes))

    # assert len(probe_uuids) == len(eval_outcomes), 'Number of probe events and pick eval events do not match'

    # pick_images = rosbag.Bag(os.path.join(pick_folder, 'pick_images.bag'))

    probe_execs = []
    probe_labels = []
    latest_probe_exec = None

    for pick_event in pick_events['events']:
        if pick_event['event_type'] == 'pick_eval':
            if pick_event['metadata']['ignore'] == 'true':
                continue
            probe_labels.append(pick_event['metadata']['eval_code'])
            probe_execs.append(latest_probe_exec)
        if pick_event['event_type'] == 'primitive_exec':
            if pick_event['metadata']['primitive_name'] == 'probe_super':
                latest_probe_exec = pick_event['metadata']['primitive_uuid']
                # item_ids.append(latest_pick_item_id)

    # probe_topics = [f'/probe_super/{uuid}' for uuid in probe_execs]

    annotations = []
    for uuid, outcome in zip(probe_execs, probe_labels):
        if outcome == 'fail_ignore':
            continue

        probe_uuid = f'{pick_uuid}_{outcome}_{uuid}'
        # print(uuid, outcome, item_id)

        # pick_images = []
        probe_images = []
        pick_images = rosbag.Bag(os.path.join(pick_folder, 'pick_images.bag'))
        for img in pick_images:
            if img.topic == f'/probe_super/{uuid}':
                cv_im = bridge.imgmsg_to_cv2(img.message)
                cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
                cv_im = cv_im[:,100:1024-100]
                cv_im = cv2.resize(cv_im, (224,224))
                probe_images.append(cv_im)    
        pick_images.close()
    
        out_vid_name = os.path.join(output_folder, f'{probe_uuid}.mp4')
        out_vid_writer = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (224, 224))
        for img in probe_images:
            # img = img.copy()
            # cv2.putText(img, 'Multipick', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out_vid_writer.write(img)
        out_vid_writer.release()

        # if args.binary:
        #     outcome_id = CLASS_STRING_TO_INT_BINARY[outcome]
        # else:
        #     outcome_id = CLASS_STRING_TO_INT[outcome]
        if outcome == 'fail_wrong_item':
            outcome = 'success'

        annotations.append((probe_uuid, outcome))

    return annotations


def main(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    for split in ['train', 'test', 'val']:
        if not os.path.exists(os.path.join(args.output, split)):
            os.makedirs(os.path.join(args.output, split))
    
    # Get list of session folders in input directory
    session_folders = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, f))]

    # For each session folder, get a list of pick folders inside it
    annotations = []
    for session_folder in session_folders:
        pick_folders = [os.path.join(session_folder, f) for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f))]
        for pick_folder in pick_folders:
            annotations.extend(process_pick_folder(pick_folder, args.output))
    
    # Split annotations into train test and eval sets
    np.random.shuffle(annotations)
    train_annotations = annotations[:int(0.8*len(annotations))]
    test_annotations = annotations[int(0.8*len(annotations)):int(0.9*len(annotations))]
    eval_annotations = annotations[int(0.9*len(annotations)):]

    for probe_uuid, outcome in train_annotations:
        if not os.path.exists(os.path.join(args.output, 'train', outcome)):
            os.makedirs(os.path.join(args.output, 'train', outcome))
        os.rename(os.path.join(args.output, f'{probe_uuid}.mp4'), os.path.join(args.output, 'train', outcome, f'{probe_uuid}.mp4'))
    
    for probe_uuid, outcome in test_annotations:
        if not os.path.exists(os.path.join(args.output, 'test', outcome)):
            os.makedirs(os.path.join(args.output, 'test', outcome))
        os.rename(os.path.join(args.output, f'{probe_uuid}.mp4'), os.path.join(args.output, 'test', outcome, f'{probe_uuid}.mp4'))
    
    for probe_uuid, outcome in eval_annotations:
        if not os.path.exists(os.path.join(args.output, 'val', outcome)):
            os.makedirs(os.path.join(args.output, 'val', outcome))
        os.rename(os.path.join(args.output, f'{probe_uuid}.mp4'), os.path.join(args.output, 'val', outcome, f'{probe_uuid}.mp4'))


    # # Write annotations to file
    # with open(os.path.join(args.output, 'train.csv'), 'w') as f:
    #     f.writelines([f'{uuid}.mp4 {outcome}\n' for uuid, outcome in train_annotations])
    # with open(os.path.join(args.output, 'test.csv'), 'w') as f:
    #     f.writelines([f'{uuid}.mp4 {outcome}\n' for uuid, outcome in test_annotations])
    # with open(os.path.join(args.output, 'val.csv'), 'w') as f:
    #     f.writelines([f'{uuid}.mp4 {outcome}\n' for uuid, outcome in eval_annotations])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to input directory')
    parser.add_argument('-o', '--output', type=str, help='Path to output directory')
    parser.add_argument('-b', '--binary', action='store_true', help='Whether to preprocess for binary classification (success/fail)')
    args = parser.parse_args()
    main(args)