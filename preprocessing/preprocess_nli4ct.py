import sys
import os
import json
import uuid
import fnmatch
import pandas as pd
from argparse import ArgumentParser


def get_evidence_list(rct_path, section_id=None):
    """
    Get the evidence list from the json file.
    Args:
        rct_path: The path to the json file.
        section_id: The id of the section to get the evidence from.

    Returns: The evidence list.
    """
    evidence_file = open(rct_path, 'r')
    evidence_data = json.load(evidence_file)
    evidence_file.close()
    del evidence_file
    if section_id is not None:
        return evidence_data[section_id]
    return evidence_data


def get_nli4ct(filename, test=False):
    evidence_struct = dict(iid=list(),
                           rct=list(),
                           trial=list(),
                           itype=list(),
                           hypotheses=list(),
                           premises=list(),
                           is_evidence_section=list(),
                           premise_section=list(),
                           evidence_section=list(),
                           order_=list(),
                           evidence_label=list(),
                           two_labeled_class=list(),
                           three_labeled_class=list())

    sections = ['Intervention', 'Eligibility', 'Results', 'Adverse Events']

    ct_path = os.path.join('../datasets/raw/Complete_dataset/CT json')
    with open(filename, 'r') as jsonfile:
        file_data = json.load(jsonfile)
        for iid, instance in file_data.items():
            for _kid, _kids in [('Primary_id', 'Primary_evidence_index'), ('Secondary_id', 'Secondary_evidence_index')]:
                if _kid in instance.keys():
                    for section in sections:
                        is_evidence_section = section == instance['Section_id']
                        candidate_evidence = get_evidence_list(os.path.join(ct_path, instance[_kid] + '.json'), section)
                        for i, evidence in enumerate(candidate_evidence):
                            evidence_struct['iid'].append(iid)
                            evidence_struct['rct'].append(instance[_kid])
                            evidence_struct['trial'].append(_kid.split('_')[0])
                            evidence_struct['itype'].append(instance['Type'])
                            evidence_struct['premise_section'].append(section)
                            evidence_struct['evidence_section'].append(instance['Section_id'])
                            evidence_struct['premises'].append(evidence)
                            evidence_struct['hypotheses'].append(instance['Statement'])
                            evidence_struct['order_'].append(i)
                            evidence_struct['is_evidence_section'].append(is_evidence_section)
                            if not test:
                                is_evidence = int(is_evidence_section and i in instance[_kids])

                                evidence_struct['evidence_label'].append(is_evidence)
                                evidence_struct['two_labeled_class'].append(instance['Label'].lower())
                                evidence_struct['three_labeled_class'].append(
                                    instance['Label'].lower() if is_evidence else 'neutral')
                            else:
                                evidence_struct['evidence_label'].append('')
                                evidence_struct['two_labeled_class'].append('')
                                evidence_struct['three_labeled_class'].append('')

    return pd.DataFrame(evidence_struct)


def prepare_two_labeled_dataset(nli4ct_df, test=False):

    def append_instance(df, __uuid, row, __test=False):
        df['uuid'].append(__uuid)
        df['iid'].append(row.iid)
        df['itype'].append(row.itype)
        df['rct'].append(row.rct)
        df['trial'].append(row.trial)
        df['premise'].append(row.premises)
        df['hypothesis'].append(row.hypotheses)
        df['order_'].append(row.order_)
        if not __test:
            df['class_label'].append(row.two_labeled_class)
        else:
            df['class_label'].append(None)

    common_criteria = '(evidence_label == 1)'
    criteria_ = "(rct == @rct_ & evidence_section == @section_a & two_labeled_class != @label_a & order_ == @order_a)"
    query = "%s & %s" % (common_criteria, criteria_)

    nli4ct_instances = dict(uuid=list(), iid=list(), itype=list(), rct=list(), trial=list(), premise=list(),
                            hypothesis=list(), order_=list(), class_label=list())
    no_neg_nli4ct_instances = dict(uuid=list(), iid=list(), itype=list(), rct=list(), trial=list(), premise=list(),
                                   hypothesis=list(), order_=list(), class_label=list())

    for iid in nli4ct_df.iid.unique():
        uuid_ = uuid.uuid1()
        only_evidence = nli4ct_df.evidence_label == 1
        # Sample A is obtained from iid
        sample_a = nli4ct_df[(nli4ct_df.iid == iid) & only_evidence]
        section_a = sample_a.evidence_section.unique()[0]
        label_a = sample_a.three_labeled_class.unique()[0]
        for i, row_ in sample_a.iterrows():
            rct_ = row_.rct
            order_a = row_.order_
            trial_ = row_.trial
            neg_sample = nli4ct_df.query(query)
            if neg_sample.shape[0] == 1:
                # if there is only one negative sample, retrieve it
                append_instance(nli4ct_instances, uuid_, row_, __test=test)
                append_instance(nli4ct_instances, uuid_, neg_sample.iloc[0], __test=test)
            elif neg_sample.shape[0] > 1:
                # if there are more than one we prioritize instances with the same value for the `trial`
                # column (Primary|Secondary)
                append_instance(nli4ct_instances, uuid_, row_, __test=test)
                append_instance(nli4ct_instances, uuid_, neg_sample.query('trial == @row_.trial').iloc[0], __test=test)
            else:
                # otherwise we save instances with no negative samples in a separated structure
                append_instance(no_neg_nli4ct_instances, uuid_, row_, __test=test)

    nli4ct_instances_df, no_neg_instances_df = [pd.DataFrame(nli4ct_instances), pd.DataFrame(no_neg_nli4ct_instances)]
    del nli4ct_instances, no_neg_nli4ct_instances
    return nli4ct_instances_df, no_neg_instances_df


def preprocess_data(input_dir, target_dir):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_file, test_file, dev_file = None, None, None
    for file_ in os.listdir(input_dir):
        if fnmatch.fnmatch(file_, '*train*'):
            train_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*test*'):
            test_file = os.path.join(input_dir, file_)
        elif fnmatch.fnmatch(file_, '*dev*'):
            dev_file = os.path.join(input_dir, file_)

    nli4ct_train_df, invalid_nli4ct_train_df = prepare_two_labeled_dataset(get_nli4ct(train_file))
    nli4ct_val_df, invalid_nli4ct_val_df = prepare_two_labeled_dataset(get_nli4ct(dev_file))
    # nli4ct_test_df = get_nli4ct(test_file, test=True)
    nli4ct_train_df.to_pickle(os.path.join(target_dir, 'nli4ct_2L_train.pkl'))
    nli4ct_val_df.to_pickle(os.path.join(target_dir, 'nli4ct_2L_val.pkl'))
    invalid_nli4ct_train_df.to_pickle(os.path.join(target_dir, 'invalid_nli4ct_2L_train.pkl'))
    invalid_nli4ct_val_df.to_pickle(os.path.join(target_dir, 'invalid_nli4ct_2L_val.pkl'))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.join('../datasets/raw/Complete_dataset'),
                        help='Path to the directory containing the raw NLI4CT dataset.')
    parser.add_argument('--target_dir', type=str, default=os.path.join('../datasets/preprocessed/nli4ct'),
                        help='Path to the directory where the preprocessed NLI4CT dataset will be stored.')

    __args = parser.parse_args()

    if not os.path.exists(__args.target_dir):
        os.makedirs(__args.target_dir)

    return __args


if __name__ == '__main__':
    args = get_args()
    preprocess_data(args.input_dir, args.target_dir)
    sys.exit(0)
