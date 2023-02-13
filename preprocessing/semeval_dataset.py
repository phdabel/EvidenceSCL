import os
import json
import numpy as np
import pandas as pd
from collections import namedtuple
from torch.utils.data import TensorDataset

import torch

CT_FOLDER = os.path.join('.', 'datasets', 'training_data', 'CT json')


def get_segment_points(feature: torch.Tensor, eos_token_id, sep_token_id):
    seg_begining_1_idx = 0
    seg_ending_idxs = (feature == eos_token_id).nonzero().flatten().detach().numpy()
    seg_separator_idxs = (feature == sep_token_id).nonzero().flatten().detach().numpy()

    seg_ending_1_idx = seg_ending_idxs[0]
    seg_begining_2_idx = seg_ending_idxs[1]
    seg_ending_2_idx = seg_ending_idxs[2]

    return seg_begining_1_idx, seg_ending_1_idx, seg_begining_2_idx, seg_ending_2_idx


def get_token_type_ids(features: torch.Tensor, eos_token_id, sep_token_id, max_length=128):
    all_token_type_ids = []
    for row, feature in enumerate(features):
        seg_begining_1_idx, seg_ending_1_idx, seg_begining_2_idx, seg_ending_2_idx = get_segment_points(feature,
                                                                                                        eos_token_id,
                                                                                                        sep_token_id)
        pair_attention_ = (seg_ending_1_idx - (seg_begining_1_idx - 1)) * [0] + (
                    seg_ending_2_idx - (seg_begining_2_idx - 1)) * [1]
        padding_ = (max_length - len(pair_attention_)) * [0]

        all_token_type_ids.append(pair_attention_ + padding_)

    return torch.tensor(all_token_type_ids)


def get_evidences(rct_filepath, section_id):
    evidence_file = open(os.path.join(CT_FOLDER, rct_filepath + '.json'), 'r')
    evidences = json.load(evidence_file)
    evidence_file.close()
    return evidences[section_id]


def get_dataset_from_dataframe(data, tokenizer, args, classdict=None, max_length=128, semeval_only=False):
    # default is binary problem

    if classdict is None:
        classdict = {'contradiction': 0, 'entailment': 1}

    if semeval_only:
        data = data[data.valid_section == True]

    inputs = tokenizer.batch_encode_plus(
        [(sample.premises, sample.hypotheses) for i, sample in data.iterrows()],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt')
    # if not train:
    # all_evidence_labels = torch.tensor([sample.evidence_label for i, sample in data.iterrows()], dtype=torch.long)
    if args.dataset != 'DATASET_EVIDENCES':
        all_class_labels = torch.tensor([classdict[sample.class_label] for i, sample in data.iterrows()],
                                        dtype=torch.long)
    else:
        all_class_labels = torch.tensor([sample.class_label for i, sample in data.iterrows()], dtype=torch.long)

    all_ids = torch.tensor([i for i, sample in data.iterrows()], dtype=torch.long)
    all_iids = [sample.iid for i, sample in data.iterrows()]

    all_token_type_ids = get_token_type_ids(inputs['input_ids'],
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            max_length=max_length)
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            all_class_labels,
                            all_ids), all_iids

    return dataset


def get_balanced_dataset_three_labels(data, tokenizer, max_length=128):
    # agrupa por rótulo de classe
    g_data = data.groupby('class_label')
    grouped_data = g_data.apply(lambda x: x.sample(g_data.size().min()))

    def map_index(_grouped_data):
        # maps row ids per class label
        _idx_map = dict(contradiction=list(), neutral=list(), entailment=list())
        for _idx in _grouped_data.index.to_list():
            _idx_map[_idx[0]].append(_idx)

        return _idx_map

    idx_map = map_index(grouped_data)
    # monta o dataset intercalando uma instância de cada classe
    dataset_items = list()

    Sample = namedtuple('Sample', ('row', 'iid', 'rct', 'ev_order', 'premise', 'hypothesis', 'section',
                                   'valid_section', 'trial', 'itype', 'evidence_label', 'class_label'))

    classdict  = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    rclassdict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    while not len(idx_map['contradiction']) == len(idx_map['neutral']) == len(idx_map['entailment']) == 0:
        try:
            for key in idx_map.keys():
                idx = idx_map[key].pop()
                row = grouped_data[grouped_data.index == idx]
                assert len(row.iid.values) == 1, "Mais de uma linha"
                dataset_items.append(Sample(
                    row=idx[1],
                    iid=row.iid.values[0],
                    rct=row.rct.values[0],
                    ev_order=row.order_.values[0],
                    premise=row.premises.values[0],
                    hypothesis=row.hypotheses.values[0],
                    section=row.section.values[0],
                    valid_section=row.valid_section.values[0],
                    trial=row.trial.values[0],
                    itype=row.itype.values[0],
                    evidence_label=row.evidence_label.values[0],
                    class_label=classdict[row.class_label.values[0]]
                ))
        except IndexError:
            continue

    inputs = tokenizer.batch_encode_plus(
        [(sample.premise, sample.hypothesis) for idx, sample in enumerate(dataset_items)],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt')

    all_evidence_labels = torch.tensor([sample.evidence_label for idx, sample in enumerate(dataset_items)],
                                       dtype=torch.long)
    all_class_labels = torch.tensor([sample.class_label for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_ids = torch.tensor([sample.row for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_token_type_ids = get_token_type_ids(inputs['input_ids'],
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            max_length=max_length)
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            all_class_labels,
                            all_evidence_labels,
                            all_ids)

    return dataset


def get_balanced_dataset_two_labels(data, tokenizer, max_length=128, exclude_not_evidences=True):
    evidence_struct = dict(iid=list(), rct=list(), trial=list(), valid_section=list(), section=list(), itype=list(),
                           sentence1=list(), order_=list(), sentence2=list(), label_=list(), class_=list())

    Sample = namedtuple('Sample', ('row', 'iid', 'rct', 'ev_order', 'premise', 'hypothesis', 'section', 'trial',
                                   'itype', 'evidence_label', 'class_label'))

    classdict = {'contradiction': 0, 'entailment': 1}
    #rclassdict = {0: 'contradiction', 1: 'entailment'}

    not_evidences = ['Intervention', 'Eligibility', 'Adverse Events', 'Results']

    for iid, instance in data.items():
        for _kid, _kids in [('Primary_id', 'Primary_evidence_index'), ('Secondary_id', 'Secondary_evidence_index')]:
            if _kid in instance.keys():
                for section_id in not_evidences:
                    is_evidence_section = section_id == instance['Section_id']
                    candidates = get_evidences(instance[_kid], section_id)
                    for i, evidence in enumerate(candidates):
                        # is evidence section condition
                        if is_evidence_section and i in instance[_kids]:
                            evidence_struct['iid'].append(iid)
                            evidence_struct['rct'].append(instance[_kid])
                            evidence_struct['itype'].append(instance['Type'])
                            evidence_struct['section'].append(instance['Section_id'])
                            evidence_struct['sentence2'].append(instance['Statement'])
                            evidence_struct['trial'].append(_kid.split('_')[0])
                            evidence_struct['order_'].append(i)
                            evidence_struct['sentence1'].append(evidence)
                            evidence_struct['class_'].append(instance['Label'].lower())
                            evidence_struct['label_'].append(1)
                            evidence_struct['valid_section'].append(is_evidence_section)
                        # if is_evidence_section and i in instance[_kids]:
                        #     evidence_struct['order_'].append(i)
                        #     evidence_struct['sentence1'].append(evidence)
                        #     evidence_struct['class_'].append(instance['Label'].lower())
                        #     evidence_struct['label_'].append(1)
                        #     evidence_struct['valid_section'].append(is_evidence_section)
                        # elif not exclude_not_evidences:
                        #     evidence_struct['order_'].append(i)
                        #     evidence_struct['sentence1'].append(evidence)
                        #     evidence_struct['class_'].append(instance['Label'].lower())
                        #     evidence_struct['label_'].append(0)
                        #     evidence_struct['valid_section'].append(False)

    evidence_df = pd.DataFrame(evidence_struct)

    last_iid = None
    dataset_items = list()
    for row, evidence in evidence_df.sort_values(by=['iid', 'section', 'trial', 'label_', 'order_'],
                                                 ascending=[True, True, True, False, True]).iterrows():
        if evidence.label_ == 1:
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
                                        evidence_label=evidence.label_,
                                        class_label=classdict[evidence.class_]))
            true_label += 1
        elif evidence.label_ == 0 and last_iid == evidence.iid and not exclude_not_evidences:
            if neg_label is None:
                neg_label = np.ceil(true_label * .25)

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
                                            evidence_label=evidence.label_,
                                            class_label=classdict[evidence.class_]))
                neg_label -= 1
            else:
                continue

    inputs = tokenizer.batch_encode_plus([(sample.hypothesis, sample.premise) for idx, sample in enumerate(dataset_items)],
                                         add_special_tokens=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_length,
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         return_tensors='pt')

    all_evidence_labels = torch.tensor([sample.evidence_label for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_class_labels = torch.tensor([sample.class_label for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_ids = torch.tensor([sample.row for idx, sample in enumerate(dataset_items)], dtype=torch.long)
    all_token_type_ids = get_token_type_ids(inputs['input_ids'],
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            max_length=max_length)
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            all_class_labels,
                            all_evidence_labels,
                            all_ids)
    return dataset, evidence_df, dataset_items


def convert_examples_to_features(data, tokenizer, max_length=128):
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
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            inputs['token_type_ids'],
                            all_labels,
                            all_ids)
    return dataset
