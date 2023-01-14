import os
import pickle
import argparse
import fnmatch
import json
import pandas as pd
from collections import namedtuple
from torch.utils.data import TensorDataset

import torch

from data_processor import SemEvalPreprocessor
CT_FOLDER = os.path.join('..', 'datasets', 'training_data', 'CT json')


def get_evidences(rct_filepath, section_id):
    evidence_file = open(os.path.join(CT_FOLDER, rct_filepath + '.json'), 'r')
    evidences = json.load(evidence_file)
    evidence_file.close()
    return evidences[section_id]


def convert_examples_to_features(data, tokenizer, max_length=358):
    evidence_struct = dict(iid=list(), rct=list(), trial=list(), valid_section=list(), section=list(), itype=list(),
                           sentence1=list(), order_=list(), sentence2=list(), label=list(), class_=list())

    Sample = namedtuple('Sample', ('row', 'iid', 'rct', 'ev_order', 'premise', 'hypothesis', 'section', 'trial',
                                   'itype', 'valid_section', 'label'))

    labeldict = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

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


def preprocess_semeval_data(inputdir,
                           targetdir,
                           lowercase=False,
                           ignore_punctuation=False,
                           num_words=None,
                           stopwords=[],
                           labeldict={},
                           bos=None,
                           eos=None):
    """
    Preprocess the data from the SemEval 2023 Task 7 dataset so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the SemEval dataset.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    train_file = ""
    dev_file = ""
    for file_ in os.listdir(inputdir):
        if fnmatch.fnmatch(file_, "*train*"):
            train_file = file_
        elif fnmatch.fnmatch(file_, "*dev*"):
            dev_file = file_

    preprocessor = SemEvalPreprocessor(lowercase=lowercase,
                                      ignore_punctuation=ignore_punctuation,
                                      num_words=num_words,
                                      stopwords=stopwords,
                                      labeldict=labeldict,
                                      bos=bos,
                                      eos=eos)

    # -------------------- Train data preprocessing -------------------- #
    print(20 * "=", " Preprocessing train set ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20 * "=", " Preprocessing dev set ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)


if __name__ == "__main__":
    default_config = "../config/semeval2023_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the SemEval 2023 - Task 7 dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing MedNLI"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess_semeval_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"]
    )
