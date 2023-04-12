import os
import json
import re
import fnmatch
import pandas as pd
from argparse import ArgumentParser
import uuid


def get_dataframe(filename, label_list=['entailment', 'contradiction']):
    __uuid, iid, premise, hypothesis, label = [], [], [], [], []
    with open(filename, 'r', encoding='utf-8') as input_data:
        for line in input_data.readlines():
            instance = json.loads(line)

            if instance['gold_label'] not in label_list:
                continue

            __uuid.append(str(uuid.uuid4()))
            iid.append(instance['pairID'])
            premise.append(instance['sentence1'])
            hypothesis.append(instance['sentence2'])
            label.append(instance['gold_label'])

    return pd.DataFrame({'uuid': __uuid, 'iid': iid, 'premise': premise, 'hypothesis': hypothesis,
                         'class_label': label})


def __remove_artifacts(mednli_dataframe):
    pattern = "\[\*\*.*?\*\*\]"
    extra_info = dict(uuid=list(), iid=list(), hypothesis=list(), original_premise=list(), artifacts=list(),
                      spans=list(), masks=list(), class_label=list())

    def get_mask(artifact):
        """
        A very straightforward way to replace artifacts with a mask token.
        Args:
            artifact: The artifact to be replaced.

        Returns: The mask token.

        """
        if 'ospital' in artifact:
            return '<ORG>'
        elif 'ocation' in artifact:
            return '<LOC>'
        elif 'Name' in artifact or 'name' in artifact:
            return '<PERSON>'
        elif 'Date' in artifact or 'Month' in artifact or 'Year' in artifact:
            return '<DATE>'
        else:
            return '<MASK>'

    for i, row in mednli_dataframe.iterrows():
        artifacts, spans, masks = [], [], []
        for j, patt in enumerate(re.finditer(pattern, row.premise)):
            artifacts.append(patt.groups())
            spans.append(patt.span())
            masks.append(get_mask(patt.groups()))

        extra_info['uuid'].append(row['uuid'])
        extra_info['iid'].append(row['iid'])
        extra_info['hypothesis'].append(row['hypothesis'])
        extra_info['original_premise'].append(row['premise'])
        extra_info['artifacts'].append(artifacts)
        extra_info['spans'].append(spans)
        extra_info['masks'].append(masks)
        extra_info['class_label'].append(row['class_label'])

    tmp = pd.DataFrame(extra_info)
    premises = list()
    for i, row in tmp.iterrows():
        if len(row.spans) > 0:
            spans = row.spans.copy()
            spans.reverse()
            x = row.original_premise
            for j, span in enumerate(spans):
                x = x[:span[0]] + row.masks[len(row.masks) - 1 - j] + x[span[1]:]
            premises.append(x)
        else:
            premises.append(row.original_premise)
    tmp['premise'] = premises
    return tmp


def prepare_two_labeled_dataset(filename):
    return __remove_artifacts(get_dataframe(filename))


def preprocess_data(input_dir, target_dir):

    train_file, test_file, dev_file = None, None, None
    for file_ in os.listdir(input_dir):
        if fnmatch.fnmatch(file_, '*train*.jsonl'):
            train_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*test*.jsonl'):
            test_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*dev*.jsonl'):
            dev_file = os.path.join(input_dir, file_)

    mednli_train_df = prepare_two_labeled_dataset(train_file)
    mednli_val_df = prepare_two_labeled_dataset(dev_file)
    mednli_test_df = prepare_two_labeled_dataset(test_file)

    mednli_train_df.to_pickle(os.path.join(target_dir, 'mednli_2L_train.pkl'))
    mednli_val_df.to_pickle(os.path.join(target_dir, 'mednli_2L_val.pkl'))
    mednli_test_df.to_pickle(os.path.join(target_dir, 'mednli_2L_test.pkl'))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.join('../datasets/raw/mednli'),
                        help='Path to the directory containing the raw MedNLI dataset.')
    parser.add_argument('--target_dir', type=str, default=os.path.join('../datasets/preprocessed/mednli'),
                        help='Path to the directory where the preprocessed MedNLI dataset will be stored.')

    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    return args


def __main__():
    args = get_args()
    preprocess_data(args.input_dir, args.target_dir)


if __name__ == '__main__':
    __main__()
