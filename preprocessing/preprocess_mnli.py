import sys
import os
import json
import fnmatch
import pandas as pd
from argparse import ArgumentParser
import uuid


def get_dataframe(filename, label_list):

    # workaround to define labels based on iid instead of gold_label
    iid_map = {'n': 'neutral', 'e': 'entailment', 'c': 'contradiction'}

    with open(filename, 'r', encoding='utf-8') as input_data:
        __uuid, iid, prompt_id, genre, hypothesis, premise, class_label = [], [], [], [], [], [], []
        for line in input_data.readlines():
            instance = json.loads(line)

            if iid_map[instance['pairID'][-1]] not in label_list:
                continue

            __uuid.append(uuid.uuid4())
            iid.append(instance['pairID'])
            prompt_id.append(instance['promptID'])
            genre.append(instance['genre'])
            hypothesis.append(instance['sentence1'])
            premise.append(instance['sentence2'])
            class_label.append(iid_map[instance['pairID'][-1]])

    return pd.DataFrame(
        {'uuid': __uuid, 'iid': iid, 'promptID': prompt_id, 'genre': genre, 'hypothesis': hypothesis,
         'premise': premise, 'class_label': class_label})\
        .sort_values(by=['iid'], ascending=[False]).reset_index(drop=True)


def __remove_incomplete_instances(dataframe, num_labels=3):
    """
    Remove instances that do not have all the labels.
    Args:
        dataframe:
        num_labels:

    Returns:

    """
    tmp_agg = dataframe.groupby('promptID').aggregate(list)
    tmp_agg['prompts_by_class'] = [len(row['class_label']) for i, row in tmp_agg.iterrows()]
    [dataframe.drop(dataframe[dataframe.uuid.isin(row.uuid)].index.tolist(), inplace=True)
     for i, row in tmp_agg[tmp_agg.prompts_by_class != num_labels].iterrows()]
    del tmp_agg
    return True


def preprocess_data(input_dir, target_dir, label_list):
    """
    Preprocess the MultiNLI dataset.
    Args:
        input_dir:
        target_dir:
        label_list:

    Returns:

    """

    train_file, val_file, test_file = None, None, None
    for file_ in os.listdir(input_dir):
        if fnmatch.fnmatch(file_, '*train*.jsonl'):
            train_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*dev_matched*.jsonl'):
            val_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*dev_mismatched*.jsonl'):
            test_file = os.path.join(input_dir, file_)

    multi_nli_train_df = get_dataframe(train_file, label_list)
    __remove_incomplete_instances(multi_nli_train_df, num_labels=len(label_list))
    multi_nli_val_df = get_dataframe(val_file, label_list)
    __remove_incomplete_instances(multi_nli_val_df, num_labels=len(label_list))
    multi_nli_test_df = get_dataframe(test_file, label_list)
    __remove_incomplete_instances(multi_nli_test_df, num_labels=len(label_list))

    multi_nli_train_df.to_pickle(os.path.join(target_dir, 'multi_nli_%dL_train.pkl' % len(label_list)))
    multi_nli_val_df.to_pickle(os.path.join(target_dir, 'multi_nli_%dL_val.pkl' % len(label_list)))
    multi_nli_test_df.to_pickle(os.path.join(target_dir, 'multi_nli_%dL_test.pkl' % len(label_list)))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.join('../datasets/raw/multinli_1.0'),
                        help='Path to the directory containing the MultiNLI dataset. (default: %(default)s)')
    parser.add_argument('--target_dir', type=str, default=os.path.join('../datasets/preprocessed/multi_nli'),
                        help='Path to the directory where the preprocessed data will be stored. (default: %(default)s)')
    parser.add_argument('--label_list', default='neutral,entailment,contradiction',
                        type=lambda s: [item for item in s.split(',')],
                        help='List of labels to be considered for the classification task. '
                             '(Default: neutral, entailment, contradiction)')

    __args = parser.parse_args()

    if not os.path.exists(__args.target_dir):
        os.makedirs(__args.target_dir)

    return __args


if __name__ == '__main__':
    args = get_args()

    print(20 * "=", "Preprocessing Dataset:", 20 * '=')
    print("Dataset name: MultiNLI")
    print("Numer of labels: %d" % len(args.label_list))

    preprocess_data(args.input_dir, args.target_dir, args.label_list)
    sys.exit(0)
