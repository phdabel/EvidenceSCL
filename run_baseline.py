import os
import random
import time
import warnings
import argparse
import torch
import torch.backends.cudnn as cudnn

from util import get_dataframes, get_dataset_from_dataframe, save_model
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier

from pipeline.biomed_roberta_baseline import train as train_roberta, validate as validate_roberta
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # models dataset
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
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')
    parser.add_argument('--ckpt', type=str, default='', help="Path to the pre-trained models")
    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.model_name = '{}_{}L_lr_{}_w_decay_{}_bsz_{}_temp_{}'. \
        format(args.model_name, args.num_classes, args.learning_rate,
               args.weight_decay, args.batch_size, args.temp)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def main_worker(args):

    best_acc = None
    classifier = LinearClassifier(RobertaModel.from_pretrained("allenai/biomed_roberta_base"),
                                  num_classes=args.num_classes)
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        classifier = torch.nn.DataParallel(classifier).cuda()

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss().cuda()
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # Obtain dataloaders
    train_df, val_df, test_df = get_dataframes(args.dataset, args.data_folder, args.num_classes)
    train_dataset, _, train_iids = get_dataset_from_dataframe(train_df, tokenizer, args.max_seq_length)
    validate_dataset, _, validate_iids = get_dataset_from_dataframe(val_df, tokenizer, args.max_seq_length)
    test_dataset = None
    if test_df is not None:
        test_dataset, _, test_iids = get_dataset_from_dataframe(test_df, tokenizer, args.max_seq_length)

    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    validation_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    for epoch in range(args.epochs):
        time1 = time.time()
        loss, train_acc = train_roberta(training_loader, classifier, criterion, optimizer, scheduler, epoch, args)
        time2 = time.time()
        print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), loss))

        val_time1 = time.time()
        validation_loss, val_acc = validate_roberta(validation_loader, classifier, criterion, epoch, args)
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (val_time2 - val_time1),
                                                                           validation_loss))

        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            print("New best accuracy: {:.3f}".format(best_acc.item()))
            save_file = os.path.join(args.save_folder, 'classifier_best.pth')
            save_model(classifier, optimizer, args, epoch, best_acc, save_file)
        print("Best accuracy: {:.3f}".format(best_acc.item()))


if __name__ == '__main__':
    __args = parse_option()

    if __args.seed is not None:
        random.seed(__args.seed)
        torch.manual_seed(__args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting'
                      ' from checkpoints.')

    main_worker(__args)
