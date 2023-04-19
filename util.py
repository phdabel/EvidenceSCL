import os
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model and dataset
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--num_classes", default=3, type=int,
                        help="The number of labels for the classifier.")
    parser.add_argument('--model_name', type=str, default='EvidenceSCL', choices=['EvidenceSCL', 'PairSCL',
                                                                                  'BioMedRoBERTa'],
                        help='Model name (default: EvidenceSCL)')
    parser.add_argument('--dataset', type=str, default='NLI4CT', choices=['NLI4CT', 'MEDNLI', 'MultiNLI'],
                        help='Dataset name (default: NLI4CT)')
    parser.add_argument('--data_folder', type=str, default='./datasets/preprocessed',
                        help='Datasets base path (default: ./datasets/preprocessed)')
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Path to the pre-trained encoder checkpoint (default: None)')

    # training
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='Number of total epochs to run')
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

    # parameters
    parser.add_argument('--alpha', type=float, default=1.,
                        help='Alpha parameter for training objective (SCL vs. CE)')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--ckpt', type=str, default=None, help="Path to the pre-trained models")
    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}L_lr_{}_w_decay_{}_bsz_{}_temp_{}'. \
        format(args.model_name, args.num_classes, args.learning_rate,
               args.weight_decay, args.batch_size, args.temp)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def get_dataframes(dataset, data_folder, num_classes):
    train_df, val_df, test_df = None, None, None
    if dataset == 'NLI4CT':
        # NLI4CT dataset uses 2 labels even though the model has 3 classes
        train_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_train.pkl"))
        val_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_val.pkl"))
        # pending item - add test_df
    elif dataset == 'MEDNLI':
        train_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_train.pkl" % num_classes))
        val_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_val.pkl" % num_classes))
        test_df = pd.read_pickle(os.path.join(data_folder, 'mednli', "mednli_%dL_test.pkl" % num_classes))
    elif dataset == 'MultiNLI':
        train_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                               "multi_nli_%dL_train.pkl" % num_classes))
        val_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                             "multi_nli_%dL_val.pkl" % num_classes))
        test_df = pd.read_pickle(os.path.join(data_folder, 'multi_nli',
                                              "multi_nli_%dL_test.pkl" % num_classes))
    return train_df, val_df, test_df


def get_segment_points(feature: torch.Tensor, eos_token_id):
    seg_beginning_1_idx = 0
    seg_ending_idx = torch.tensor(feature == eos_token_id, dtype=torch.bool).nonzero().flatten().detach().numpy()
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


def get_dataset_from_dataframe(dataframe, tokenizer, max_seq_length: Optional[int] = None):
    """
    Args:
        dataframe:
        tokenizer:
        max_seq_length:

    Returns: dataset, all_uuids, all_iid

    """
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

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

    labels = torch.tensor([class_dict[row.class_label] for _, row in dataframe.iterrows()], dtype=torch.long)
    all_iid = [row.iid for _, row in dataframe.iterrows()]
    all_uuids = [row.uuid for _, row in dataframe.iterrows()]
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            labels)

    return dataset, all_uuids, all_iid


def get_dataloaders(dataset, data_folder, tokenizer, batch_size, workers, max_seq_length, num_classes):
    # Obtain dataloaders
    train_df, val_df, test_df = get_dataframes(dataset, data_folder, num_classes)
    train_dataset, _, train_iids = get_dataset_from_dataframe(train_df, tokenizer, max_seq_length)
    validate_dataset, _, validate_iids = get_dataset_from_dataframe(val_df, tokenizer, max_seq_length)
    test_dataset, test_iids = None, None
    if test_df is not None:
        test_dataset, _, test_iids = get_dataset_from_dataframe(test_df, tokenizer, max_seq_length)

    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, pin_memory=True)
    validation_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=workers, pin_memory=True)
    test_loader = None if test_dataset is None else DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                               num_workers=workers, pin_memory=True)

    return training_loader, validation_loader, test_loader, train_iids, validate_iids, test_iids


def save_model(model, optimizer, args, epoch, best_acc, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
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
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range =\
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index
