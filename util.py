import os
import json

import pandas as pd
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional
from statistics import mode
from sklearn.metrics import accuracy_score
from zipfile import ZipFile


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model and dataset
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--num_classes", default=3, type=int,
                        help="The number of labels for the classifier.")
    parser.add_argument('--model_name', type=str, default='evidencescl', choices=['evidencescl', 'pairscl',
                                                                                  'biomed'],
                        help='Model name (default: evidencescl)')
    parser.add_argument('--evidence_retrieval', action='store_true', default=False,
                        help='Use evidence retrieval model to select the evidence sentences.')
    parser.add_argument('--dataset', type=str, default='nli4ct', choices=['nli4ct', 'mednli', 'multinli', 'local'],
                        help='Dataset name (default: nli4ct)')
    parser.add_argument('--dataset_suffix', type=str, default=None,
                        help="Extra information to be added in the dataset's name (default: None). "
                             "This is useful when combining datasets. For example, if you want to combine "
                             "the datasets 'nli4ct' on a pre-trained checkpoint trained on 'mednli', you can set "
                             "the parameters as if you was training on nli4ct and add 'mednli' in the --dataset_suffix "
                             "to concatenate it at the end of the folder's name.")
    parser.add_argument('--combine', action='store_true', default=False,
                        help='Combine dataset for training. This option should should be used together with a '
                             'pre-trained checkpoint from another dataset and a dataset suffix. For example: '
                             'to combine NLI4CT on a model trained on MedNLI, you should to inform the path of the '
                             'MedNLI checkpoint with the --encoder_ckpt argument, set the --resume option, '
                             '--dataset nli4ct, --dataset_suffix mednli, and inform the remaining parameters as usual.')
    parser.add_argument('--data_folder', type=str, default='./datasets/preprocessed',
                        help='Datasets base path (default: ./datasets/preprocessed)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from the checkpoint. If no checkpoint is informed try to obtain it '
                             'from parameters settings.')
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Path to the pre-trained encoder checkpoint (default: None)')

    # evaluation
    parser.add_argument('--evaluate_dataset', type=str, default=None, choices=['nli4ct', 'mednli', 'multinli'],
                        help="Dataset name to evaluate the model (default: None). "
                             "If None, the dataset will be the same as the training dataset.")
    parser.add_argument('--evaluate_stage', type=str, default='test', choices=['training', 'validation', 'test'],
                        help="Informs dataset's stage that will be used for evaluation (default: test).")

    # training
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='Number of total epochs to run. If combining a dataset, this value will be added to the '
                             'number of epochs of the pre-trained model. (default: 3)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size in number of sentences per batch')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--l1_regularization', type=float, default=0.1,
                        help='Coefficient for L1 Regularization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save frequency')

    # distributed training
    parser.add_argument('--rank', type=int, default=-1,
                        help='node rank for distributed training on gpus (default: -1)')
    parser.add_argument('--world_size', type=int, default=-1,
                        help='world_size for distributed training on gpus (default: -1 means using all available gpus)')
    parser.add_argument('--dist_url', type=str, default='env://',
                        help='url used to set up distributed training (default: env://)')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend (default: nccl)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for distributed training on gpus (default: -1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch N processes per node, which has N '
                             'gpus. This is the fastest way to use PyTorch for either single node or multi node data '
                             'parallel training')

    # parameters
    parser.add_argument('--alpha', type=float, default=1.,
                        help='Alpha parameter for training objective (SCL vs. CE)')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--ckpt', type=str, default=None, help="Path to the pre-trained models")
    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset) if args.dataset_suffix is None \
        else './save/{}_{}_models'.format(args.dataset, args.dataset_suffix)
    args.model_name = '{}_{}L_len_{}_lr_{}_w_decay_{}_bsz_{}_temp_{}{}'. \
        format(args.model_name, args.num_classes, args.max_seq_length,
               args.learning_rate, args.weight_decay, args.batch_size, args.temp,
               '_er' if args.evidence_retrieval else '')

    args.start_epoch = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.combine and not args.encoder_ckpt:
        raise ValueError("When combining datasets, the encoder checkpoint from the pre-trained model must be informed.")

    if args.combine and not args.dataset_suffix:
        raise ValueError("When combining datasets, the dataset suffix from the pre-trained model must be informed.")

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def compute_real_accuracy(results, args, stage, unlabeled=False):
    """
    Compute accuracy for SemEval-2023 Task 7.

    Group the results by the instance id (iid) and trial and computes the accuracy based on the majority label or
        the presence of at least one entailment label within the predictions.

    Args:
        results: dictionary
        args: argparse.Namespace
        stage: string
        unlabeled: boolean (if True, the gold_label must be present in the results)

    Returns:
        maj_acc: float (accuracy based on the majority label by iid)
        alo_acc: float (accuracy based on the presence of at least one entailment label by iid)

    """
    keys_to_remove = [key for key in results.keys() if len(results[key]) == 0]
    for key in keys_to_remove:
        del results[key]

    if not unlabeled and 'gold_label' not in results.keys():
        raise ValueError("gold_label not found in results")

    results_df = pd.DataFrame(results)
    _iids = results_df.iid.unique().tolist()

    def get_itype(iid):
        return results_df[results_df.iid == iid].itype.tolist()[0]

    # remove neutral labels
    results_df.drop([i for i, _ in results_df[results_df.predicted_label == 2].iterrows()], inplace=True)
    maj_acc, alo_acc = None, None
    grouped_results = results_df.groupby(['iid', 'trial']).aggregate(list)
    struct_maj = dict()
    struct_alo = dict()
    struct_glb = dict()
    grouped_results['majority_label'] = [mode(row.predicted_label) for _, row in grouped_results.iterrows()]
    grouped_results['at_least_one'] = [int(sum(row.predicted_label) > 0) for _, row in grouped_results.iterrows()]

    for _iid in _iids:
        maj_ = grouped_results[grouped_results.index == (_iid, 'Primary')].majority_label.tolist()[0]
        alo_ = grouped_results[grouped_results.index == (_iid, 'Primary')].at_least_one.tolist()[0]
        if get_itype(_iid) == 'Comparison':
            maj_ = int(maj_ == 1 and grouped_results[grouped_results.index == (_iid, 'Secondary')]
                       .majority_label.tolist()[0] == 1)
            alo_ = int(alo_ == 1 and grouped_results[grouped_results.index == (_iid, 'Secondary')]
                       .at_least_one.tolist()[0] == 1)

        struct_maj[_iid] = maj_
        struct_alo[_iid] = alo_
        if not unlabeled:
            struct_glb[_iid] = mode(grouped_results[grouped_results.index == (_iid, 'Primary')].gold_label.tolist()[0])

    if not unlabeled:
        maj_acc = accuracy_score(list(struct_glb.values()), list(struct_maj.values()))
        alo_acc = accuracy_score(list(struct_glb.values()), list(struct_alo.values()))

    generate_results_file(struct_maj, args, 'maj_{}_'.format(stage))
    generate_results_file(struct_alo, args, 'alo_{}_'.format(stage))

    return maj_acc, alo_acc, grouped_results


def generate_results_file(struct, args, prefix):
    filename_ = 'results.json'
    for key in struct.keys():
        struct[key] = {'Prediction': 'Contradiction' if struct[key] == 0 else 'Entailment'}

    with open(filename_, 'w') as file_:
        json.dump(struct, file_, indent=4)

    output_file = os.path.join(args.save_folder, prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.zip')
    with ZipFile(output_file, 'w') as zipObj:
        zipObj.write(filename_)

    os.unlink(filename_)
    print("Files %s create inside of the file: %s." % (filename_, output_file))


def build_evaluation_file(results, args):

    filename_ = 'results.json'

    keys_to_remove = [key for key in results.keys() if len(results[key]) == 0]
    for key in keys_to_remove:
        del results[key]

    if 'gold_label' not in results.keys():
        raise ValueError("gold_label not found in results")
    
    predicted_labels = results['predicted_label']
    gold_labels = results['gold_label']

    results_df = pd.DataFrame(results)
    _iids = results_df.iid.unique().tolist()

    order_combined_df = results_df.groupby(['iid', 'predicted_label', 'trial'])['order_'].agg(list)
    grouped_results = results_df.groupby(['iid', 'trial']).agg(list)

    res = {}
    for _iid in _iids:
        if _get_itype(_iid, results_df) == 'Comparison':
            if _evidence_exists(_iid, results_df):
                primary_response = filter_order(order_combined_df[_iid][1]['Primary']) if _primary_key_exists(_iid, results_df) else []
                secondary_response = filter_order(order_combined_df[_iid][1]['Secondary']) if _secondary_key_exists(_iid, results_df) else []
            else:
                primary_response = []
                secondary_response = []
                res[_iid] = {'Primary_evidence_index': primary_response, 'Secondary_evidence_index': secondary_response}
        else:
            if _evidence_exists(_iid, results_df):
                primary_response = filter_order(order_combined_df[_iid][1]['Primary']) if _primary_key_exists(_iid, results_df) else []
            else:
                primary_response = []

        res[_iid] = {'Primary_evidence_index': primary_response}

    acc = accuracy_score(predicted_labels, gold_labels)

    with open(filename_, 'w') as file_:
        json.dump(res, file_, indent=4)

    output_file = os.path.join(args.save_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.zip')
    with ZipFile(output_file, 'w') as zipObj:
        zipObj.write(filename_)
    
    os.unlink(filename_)
    print("Files %s create inside of the file: %s." % (filename_, output_file))

    return grouped_results, acc

def filter_order(order_list):
  filtered_list = []
  order_ = order_list
  for i, value in enumerate(order_):
    if value <= order_[i-1] and i > 0:
      return filtered_list
    else:
      filtered_list.append(value)
  return filtered_list

def _evidence_exists(iid, results_df):
  return any(results_df[results_df.iid == iid].predicted_label.tolist()) == 1

def _get_itype(iid, results_df):
  return results_df[results_df.iid == iid].itype.tolist()[0]

def _primary_key_exists(iid, results_df):
  return 'Primary' in results_df[results_df.iid == iid][results_df.predicted_label == 1].trial.tolist()

def _secondary_key_exists(iid, results_df):
  return 'Secondary' in results_df[results_df.iid == iid][results_df.predicted_label == 1].trial.tolist()


def get_dataframes(dataset, data_folder, num_classes):
    train_df, val_df, test_df = None, None, None
    if dataset == 'nli4ct':
        # NLI4CT dataset uses 2 labels even though the model has 3 classes
        train_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_train.pkl"))
        val_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_val.pkl"))
        test_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', 'nli4ct_2L_test.pkl'))
    elif dataset == 'mednli':
        train_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_train.pkl" % num_classes))
        val_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_val.pkl" % num_classes))
        test_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_test.pkl" % num_classes))
    elif dataset == 'multinli':
        train_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                               "multi_nli_%dL_train.pkl" % num_classes))
        val_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                             "multi_nli_%dL_val.pkl" % num_classes))
        test_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                              "multi_nli_%dL_test.pkl" % num_classes))
    elif dataset == 'local':
        train_df = pd.read_pickle(os.path.join(data_folder, 'local', "train_local.pkl"))
        val_df = pd.read_pickle(os.path.join(data_folder, 'local', "dev_local.pkl"))
        test_df = pd.read_pickle(os.path.join(data_folder, 'local', "test_local.pkl"))

    return train_df, val_df, test_df


def get_segment_points(feature: torch.Tensor, eos_token_id):
    seg_beginning_1_idx = 0
    seg_ending_idx = (feature == eos_token_id).nonzero()
    seg_ending_1_idx = seg_ending_idx[0]
    seg_beginning_2_idx = seg_ending_idx[1]
    seg_ending_2_idx = seg_ending_idx[2]

    return seg_beginning_1_idx, seg_ending_1_idx, seg_beginning_2_idx, seg_ending_2_idx


def get_token_type_ids(features: torch.Tensor, eos_token_id, max_seq_length=128):
    all_token_type_ids = []
    for row, feature in enumerate(features):
        seg_beginning_1_idx, seg_ending_1_idx, seg_beginning_2_idx, seg_ending_2_idx = get_segment_points(feature,
                                                                                                          eos_token_id)
        pair_attention_ = (seg_ending_1_idx - (seg_beginning_1_idx - 1)) * [0] + (
                seg_ending_2_idx - (seg_beginning_2_idx - 1)) * [1]
        padding_ = (max_seq_length - len(pair_attention_)) * [0]

        all_token_type_ids.append(pair_attention_ + padding_)

    return torch.tensor(all_token_type_ids)


def get_dataset_from_dataframe(dataframe, tokenizer, max_seq_length: Optional[int] = None, unlabeled=False):
    """
    Args:
        dataframe:
        tokenizer:
        max_seq_length:
        unlabeled:

    Returns: dataset, all_uuids, all_iid, all_trials, all_orders
        A TensorDataset containing the input_ids, attention_mask, token_type_ids, and class and evidence labels,
        respectively. If test is True, the dataset will not contain the class and evidence labels.

        Also returns a list of generated uuids, and the original iids.
        It may bring trials (if primary or secondary) and sentence orders.
    """
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

    has_evidence_column = True if 'evidence_label' in dataframe.columns else False
    class_dict = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    inputs = tokenizer.batch_encode_plus([(row.premise, row.hypothesis) for _, row in dataframe.iterrows()],
                                         add_special_tokens=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_seq_length,
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         return_tensors='pt')

    all_token_type_ids = get_token_type_ids(inputs['input_ids'],
                                            tokenizer.eos_token_id,
                                            max_seq_length)

    class_labels = torch.tensor([class_dict[row.class_label] if not unlabeled else -1
                                 for _, row in dataframe.iterrows()], dtype=torch.long)
    evidence_labels = torch.tensor([row.evidence_label if has_evidence_column and not unlabeled else -1
                                    for _, row in dataframe.iterrows()], dtype=torch.long)

    all_iid = [row.iid for _, row in dataframe.iterrows()]
    all_uuids = [row.uuid for _, row in dataframe.iterrows()]
    all_types = [row.itype for _, row in dataframe.iterrows()] if 'itype' in dataframe.columns else None
    all_trials = [row.trial for _, row in dataframe.iterrows()] if 'trial' in dataframe.columns else None
    all_orders = [row.order_ for _, row in dataframe.iterrows()] if 'order_' in dataframe.columns else None
    all_genres = [row.genre for _, row in dataframe.iterrows()] if 'genre' in dataframe.columns else None

    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            class_labels,
                            evidence_labels)

    return dataset, all_uuids, all_iid, all_trials, all_orders, all_genres, all_types


def get_dataloaders(dataset, data_folder, tokenizer, batch_size, workers, max_seq_length, num_classes):
    """

    Args:
        dataset: str
        data_folder: str
        tokenizer: transformers.PreTrainedTokenizer
        batch_size: int
        workers: int
        max_seq_length: int
        num_classes: int

    Returns: dict
        A dictionary containing dataloaders, iids, trials, orders, and genres for train, validation and test sets.
    """
    # Obtain dataloaders
    train_df, val_df, test_df = get_dataframes(dataset, data_folder, num_classes)
    train_dataset, _, train_iids, train_trials, train_orders, train_genres, train_types = get_dataset_from_dataframe(
        train_df, tokenizer, max_seq_length)
    validate_dataset, _, validation_iids, validation_trials, validation_orders, validation_genres, validation_types = \
        get_dataset_from_dataframe(val_df, tokenizer, max_seq_length)

    test_dataset, _, test_iids, test_trials, test_orders, test_genres, test_types = \
        get_dataset_from_dataframe(test_df, tokenizer, max_seq_length, unlabeled=dataset == 'nli4ct')

    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, pin_memory=True)
    validation_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)

    return {'loader': {'training': training_loader,
                       'validation': validation_loader,
                       'test': test_loader},
            'iids': {
                'training': train_iids,
                'validation': validation_iids,
                'test': test_iids},
            'trials': {
                'training': train_trials,
                'validation': validation_trials,
                'test': test_trials},
            'orders': {
                'training': train_orders,
                'validation': validation_orders,
                'test': test_orders},
            'genres': {
                'training': train_genres,
                'validation': validation_genres,
                'test': test_genres},
            'types': {
                'training': train_types,
                'validation': validation_types,
                'test': test_types}
            }


def get_dataframe(data_folder, dataset_name, filename):
    """

    Args:
        data_folder: str
        dataset_name: str - name of the dataset [nli4ct, mednli, multinli]
        filename: str

    Returns: pd.DataFrame
        Returns the dataframe from the given filename.
    """
    return pd.read_pickle(os.path.join(data_folder, dataset_name, filename))


def get_dataloader(data_folder, dataset_name, filename, tokenizer, batch_size, workers, max_seq_length,
                   unlabeled=False):
    """

    Args:
        data_folder: str
        dataset_name: str - name of the dataset [nli4ct, mednli, multinli]
        filename: str
        tokenizer: transformers.PreTrainedTokenizer
        batch_size: int
        workers: int
        max_seq_length: int
        unlabeled: bool

    Returns: Tuple[DataLoader, List[str], List[str], List[int]]
        Returns a dataloader from the given filename and the list of original iids, trials, and
        the sentence order list.
    """
    dataset_df = get_dataframe(data_folder, dataset_name, filename)
    dataset, _, iids, trials, orders, genres, types = get_dataset_from_dataframe(dataset_df, tokenizer, max_seq_length,
                                                                                 unlabeled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return dataloader, iids, trials, orders, genres


def save_model(model, optimizer, args, epoch, best_metric, save_file):
    """
    Args:
        model: transformers.PreTrainedModel
        optimizer: torch.optim.Optimizer
        args: argparse.Namespace
        epoch: int
        best_metric: float
        save_file: str

    Returns: None
    """
    print('==> Saving...')
    state = {
        'args': args,
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric
    }
    torch.save(state, save_file)
    del state


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask, so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    __weighted_sum = weights.bmm(tensor)

    while mask.dim() < __weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(__weighted_sum).contiguous().float()

    return __weighted_sum * mask


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index = \
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range = \
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def epoch_summary(model_name, epoch, training_semeval_metric, validation_semeval_metric, best_validation_metric,
                  metric_name='accuracy'):
    """

    Args:
        model_name:
        epoch:
        training_semeval_metric:
        validation_semeval_metric:
        best_validation_metric:
        metric_name:

    Returns:

    """
    print("=========================")
    print("Model:               {}".format(model_name))
    print("Epoch:               {}".format(epoch))
    print("-------------------------")
    print("Metric name: {}".format(metric_name))
    print("Training metric:   {:.3f}".format(training_semeval_metric))
    print("Validation metric: {:.3f}".format(validation_semeval_metric))
    print("Best metric:       {:.3f}".format(best_validation_metric))
    print("=========================")
