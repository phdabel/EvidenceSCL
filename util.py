import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from transformers import InputExample, InputFeatures
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Union, Optional, List


def get_dataframes(dataset, data_folder, num_classes):
    train_df, val_df, test_df = None, None, None
    if dataset == 'NLI4CT':
        # NLI4CT dataset uses 2 labels even though the model has 3 classes
        train_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_train.pkl" % num_classes))
        val_df = pd.read_pickle(os.path.join(data_folder, 'nli4ct', "nli4ct_2L_val.pkl" % num_classes))
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
    seg_ending_idx = (feature == eos_token_id).nonzero().flatten().detach().numpy()
    seg_ending_1_idx = seg_ending_idx[0]
    seg_beginning_2_idx = seg_ending_idx[1]
    seg_ending_2_idx = seg_ending_idx[2]

    return seg_beginning_1_idx, seg_ending_1_idx, seg_beginning_2_idx, seg_ending_2_idx


def get_token_type_ids(features: torch.Tensor, eos_token_id, sep_token_id, max_seq_length=128):
    all_token_type_ids = []
    for row, feature in enumerate(features):
        seg_beginning_1_idx, seg_ending_1_idx, seg_beginning_2_idx, seg_ending_2_idx = get_segment_points(feature,
                                                                                                          eos_token_id,
                                                                                                          sep_token_id)
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
                                            tokenizer.sep_token_id,
                                            max_seq_length)

    labels = torch.tensor([class_dict[row.class_label] for _, row in dataframe.iterrows()], dtype=torch.long)
    all_iid = [row.iid for _, row in dataframe.iterrows()]
    all_uuids = [row.uuid for _, row in dataframe.iterrows()]
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            all_token_type_ids,
                            labels)

    return dataset, all_uuids, all_iid


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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.short_name = name.lower().replace(' ', '_')
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __csv__(self):
        sep = ';'
        fmtstr = sep.join(['{short_name}',
                           '{val' + self.fmt + '}'])
        return fmtstr.format(**self.__dict__).split(sep)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logfile=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile
        self.header = False

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def log_metrics(self, batch, sep=";"):
        epoch = int(self.prefix[8:-1])
        entries = dict()
        entries['epoch'] = epoch
        entries['batch'] = batch
        for i, meter in enumerate(self.meters):
            metric, value = meter.__csv__()
            entries[metric] = float(value)

        _logfile = open(self.logfile, 'a')
        if not self.header and epoch == 0:
            self.header = sep.join([key for key in entries.keys()]) + '\n'
            _logfile.write(self.header)
        content = sep.join([str(entries[key]) for key in entries.keys()]) + '\n'
        _logfile.write(content)
        _logfile.close()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def convert_examples_to_features(examples: Union[List[InputExample], "tf.data.Dataset"],
                                 tokenizer: PreTrainedTokenizer,
                                 max_length: Optional[int] = None,
                                 label_list=None,
                                 output_mode=None):
    if max_length is None:
        max_length = tokenizer.model_max_length

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == 'classification':
            return label_map[example.label]
        elif output_mode == 'regression':
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer([(example.text_a, example.text_b) for example in examples],
                               max_length=max_length,
                               padding='max_length',
                               truncation=True,
                               return_token_type_ids=True)

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


def load_and_cache_examples(args, processor, tokenizer, evaluate, dataset):
    print("begin convert data")
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_folder,
        "cached_{}_{}_{}_{}".format(
            evaluate,
            args.model,
            str(args.max_seq_length),
            dataset
        ),
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
        if evaluate == "test_match" or evaluate == "test_mismatch":
            all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guid)
        else:
            # print(features[0])
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        label_list = processor.get_labels()
        if evaluate == "test_match" or evaluate == "test_mismatch":
            examples = processor.get_examples()
            features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode="classification"
            )
            torch.save(features, cached_features_file)
            all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guid)
        else:
            examples = processor.get_examples()
            features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode="classification"
            )
            torch.save(features, cached_features_file)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    print("finish build dataset")
    return dataset


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

    # Reshape the mask so it matches the size of the input tensor.
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
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


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
