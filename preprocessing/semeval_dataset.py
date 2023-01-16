import os
import json
import numpy as np
import pandas as pd
from collections import namedtuple
from torch.utils.data import TensorDataset

import torch

CT_FOLDER = os.path.join('.', 'datasets', 'training_data', 'CT json')


def get_evidences(rct_filepath, section_id):
    evidence_file = open(os.path.join(CT_FOLDER, rct_filepath + '.json'), 'r')
    evidences = json.load(evidence_file)
    evidence_file.close()
    return evidences[section_id]


def convert_examples_to_features_balanced_dataset(data, tokenizer, max_length=358):
    evidence_struct = dict(iid=list(), rct=list(), trial=list(), valid_section=list(), section=list(), itype=list(),
                           sentence1=list(), order_=list(), sentence2=list(), label=list(), class_=list())

    Sample = namedtuple('Sample', ('row', 'iid', 'rct', 'ev_order', 'premise', 'hypothesis', 'section', 'trial',
                                   'itype', 'label'))

    labeldict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    not_evidences = ['Intervention', 'Eligibility', 'Adverse Events', 'Results']

    for iid, instance in data.items():
        for _kid, _kids in [('Primary_id', 'Primary_evidence_index'), ('Secondary_id', 'Secondary_evidence_index')]:
            if _kid in instance.keys():
                for section_id in not_evidences:
                    is_evidence = section_id == instance['Section_id']
                    candidates = get_evidences(instance[_kid], section_id)
                    for i, evidence in enumerate(candidates):
                        # is evidence section condition
                        evidence_struct['iid'].append(iid)
                        evidence_struct['rct'].append(instance[_kid])
                        evidence_struct['itype'].append(instance['Type'])
                        evidence_struct['section'].append(instance['Section_id'])
                        evidence_struct['sentence2'].append(instance['Statement'])
                        evidence_struct['trial'].append(_kid.split('_')[0])
                        if is_evidence and i in instance[_kids]:
                            evidence_struct['order_'].append(i)
                            evidence_struct['sentence1'].append(evidence)
                            evidence_struct['class_'].append(instance['Label'].lower())
                            evidence_struct['label'].append(1)
                            evidence_struct['valid_section'].append(is_evidence)
                        else:
                            evidence_struct['order_'].append(i)
                            evidence_struct['sentence1'].append(evidence)
                            evidence_struct['class_'].append('neutral')
                            evidence_struct['label'].append(-1)
                            evidence_struct['valid_section'].append(False)

    evidence_df = pd.DataFrame(evidence_struct)

    last_iid = None
    dataset_items = list()
    for row, evidence in evidence_df.sort_values(by=['iid', 'section', 'trial', 'label', 'order_'],
                                                 ascending=[True, True, True, False, True]).iterrows():
        if evidence.label == 1:
            # reset for new iid
            if last_iid is None or evidence.iid != last_iid:
                true_label = 0
                neg_label = None
                last_iid = evidence.iid

            dataset_items.append(Sample(row=row,
                                        iid=evidence.iid,
                                        rct=evidence.rct,
                                        ev_order=evidence.order_,
                                        premise=evidence.sentence1,
                                        hypothesis=evidence.sentence2,
                                        section=evidence.section,
                                        trial=evidence.trial,
                                        itype=evidence.itype,
                                        label=labeldict[evidence.class_]))
            true_label += 1
        elif evidence.label == -1 and last_iid == evidence.iid:
            if neg_label is None:
                neg_label = np.ceil(true_label * 0.5)

            if neg_label > 0:
                dataset_items.append(Sample(row=row,
                                            iid=evidence.iid,
                                            rct=evidence.rct,
                                            ev_order=evidence.order_,
                                            premise=evidence.sentence1,
                                            hypothesis=evidence.sentence2,
                                            section=evidence.section,
                                            trial=evidence.trial,
                                            itype=evidence.itype,
                                            label=labeldict[evidence.class_]))
                neg_label -= 1
            else:
                continue

    inputs = tokenizer.batch_encode_plus([(sample.premise, sample.hypothesis) for idx, sample in enumerate(dataset_items)],
                                         add_special_tokens=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_length,
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')

    all_labels = torch.tensor([sample.label for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_ids = torch.tensor([sample.row for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    dataset = TensorDataset(all_ids,
                            inputs['input_ids'],
                            inputs['attention_mask'],
                            inputs['token_type_ids'],
                            all_labels)
    return dataset


def convert_examples_to_features(data, tokenizer, max_length=358):
    evidence_struct = dict(iid=list(), rct=list(), trial=list(), valid_section=list(), section=list(), itype=list(),
                           sentence1=list(), order_=list(), sentence2=list(), label=list(), class_=list())

    Sample = namedtuple('Sample', ('row', 'iid', 'rct', 'ev_order', 'premise', 'hypothesis', 'section', 'trial',
                                   'itype', 'valid_section', 'label'))

    labeldict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    not_evidences = ['Intervention', 'Eligibility', 'Adverse Events', 'Results']

    for iid, instance in data.items():
        for _kid, _kids in [('Primary_id', 'Primary_evidence_index'), ('Secondary_id', 'Secondary_evidence_index')]:
            if _kid in instance.keys():
                for section_id in not_evidences:
                    is_evidence = section_id == instance['Section_id']
                    candidates = get_evidences(instance[_kid], section_id)
                    for i, evidence in enumerate(candidates):
                        # is evidence section condition
                        evidence_struct['iid'].append(iid)
                        evidence_struct['rct'].append(instance[_kid])
                        evidence_struct['itype'].append(instance['Type'])
                        evidence_struct['section'].append(instance['Section_id'])
                        evidence_struct['sentence2'].append(instance['Statement'])
                        evidence_struct['trial'].append(_kid.split('_')[0])
                        if is_evidence and i in instance[_kids]:
                            evidence_struct['order_'].append(i)
                            evidence_struct['sentence1'].append(evidence)
                            evidence_struct['class_'].append(instance['Label'].lower())
                            evidence_struct['label'].append(1)
                            evidence_struct['valid_section'].append(is_evidence)
                        else:
                            evidence_struct['order_'].append(i)
                            evidence_struct['sentence1'].append(evidence)
                            evidence_struct['class_'].append('neutral')
                            evidence_struct['label'].append(-1)
                            evidence_struct['valid_section'].append(False)

    evidence_df = pd.DataFrame(evidence_struct)

    dataset_items = list()
    for row, evidence in evidence_df[evidence_df.label == 1].sort_values(by=['iid', 'trial', 'order_'],
                                                                         ascending=[True, True, True]).iterrows():
        dataset_items.append(Sample(row=row,
                                    iid=evidence.iid,
                                    rct=evidence.rct,
                                    ev_order=evidence.order_,
                                    premise=evidence.sentence1,
                                    hypothesis=evidence.sentence2,
                                    section=evidence.section,
                                    trial=evidence.trial,
                                    itype=evidence.itype,
                                    valid_section=evidence.valid_section,
                                    label=labeldict[evidence.class_]))

    inputs = tokenizer.batch_encode_plus([(sample.premise, sample.hypothesis) for idx, sample in enumerate(dataset_items)],
                                         add_special_tokens=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_length,
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')

    all_labels = torch.tensor([sample.label for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_ids = torch.tensor([sample.row for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    dataset = TensorDataset(all_ids,
                            inputs['input_ids'],
                            inputs['attention_mask'],
                            inputs['token_type_ids'],
                            all_labels)
    return dataset
