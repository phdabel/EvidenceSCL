import os
import json
import argparse
import time
import random
import pandas as pd
import warnings
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from sklearn.metrics import accuracy_score

from preprocessing.semeval_dataset import get_balanced_dataset_three_labels, get_balanced_dataset_two_labels, \
    get_dataset_from_dataframe
from util import accuracy, save_model, AverageMeter, ProgressMeter, get_lr, EarlyStopping
from torch.utils.data import DataLoader
from bert_model import BertForCL, LinearClassifier, PairSupConBert


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--model', type=str, default='ROBERTA')
    parser.add_argument('--dataset', type=str, default='DATASET_TWO', help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets', help='path to custom dataset')

    # training
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--coefficient', type=float, default=0.1,
                        help='L1 regularization coefficient.')  # L1 regularization
    parser.add_argument('--weight_decay', type=float, default=0.4,
                        help='weight decay')  # weight decay corresponds to the L2 regularization factor
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64,
                        help='number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--log_epochs', action='store_true',
                        help='Create a CSV file to log the metrics.')

    # distribute
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch N processes per node, '
                             'which has N GPUs. This is the fastest way to use PyTorch for either single '
                             'node or multi node data parallel training')
    # parameters
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle dataloader')
    parser.add_argument('--ckpt', type=str, default='',
                        help="path to pre-trained model")
    parser.add_argument('--eta', type=float, default=1e-5,
                        help='minimum value')

    args = parser.parse_args()

    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_grad_{}_l1_coefficient_{}'.\
        format(args.dataset, args.model, args.learning_rate, args.weight_decay, args.batch_size,
               args.gradient_accumulation_steps, args.coefficient)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.log_path = os.path.join(args.data_folder, 'logs')
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)

    return args


def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting'
                      ' from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc1 = None
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = PairSupConBert(BertForCL.from_pretrained(
        "allenai/biomed_roberta_base",  # Use the 12-layer Biomed Roberta model from allenai, with a cased vocab.
        num_labels=args.max_seq_length,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ), is_train=False)

    tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

    classifier = LinearClassifier(BertForCL.from_pretrained(
        "allenai/biomed_roberta_base",  # Use the 12-layer Biomed Roberta model from allenai, with a cased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ), num_classes=2)

    state_dict = None
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt)
        state_dict = ckpt['model']

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else None
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # construct data loader
    if args.dataset == 'DATASET_EVIDENCES':
        # used to train a evidence identifier
        semeval_datafolder = os.path.join(args.data_folder, 'preprocessed', args.dataset)
        train_filename = os.path.join(semeval_datafolder, 'dataset_two_evidence_training.pkl')
        dev_filename = os.path.join(semeval_datafolder, 'dataset_two_evidence_validation.pkl')

        training_data = pd.read_pickle(train_filename)
        training_data = training_data.reset_index(drop=True)
        dev_data = pd.read_pickle(dev_filename)
        dev_data = dev_data.reset_index(drop=True)

        train_dataset, _ = get_dataset_from_dataframe(training_data,
                                                      tokenizer=tokenizer,
                                                      args=args,
                                                      max_length=args.max_seq_length)

        validation_dataset, _ = get_dataset_from_dataframe(dev_data,
                                                           tokenizer=tokenizer,
                                                           args=args,
                                                           max_length=args.max_seq_length)
    elif args.dataset == 'DATASET_TWO':

        semeval_datafolder = os.path.join(args.data_folder, 'preprocessed', args.dataset)
        train_filename = os.path.join(semeval_datafolder, 'dataset_two_combined_training.pkl')
        dev_filename = os.path.join(semeval_datafolder, 'dataset_two_combined_validation.pkl')
        semeval_filename = os.path.join(semeval_datafolder, 'dataset_two_semeval_validation.pkl')

        training_data = pd.read_pickle(train_filename)
        training_data = training_data.reset_index(drop=True)
        dev_data = pd.read_pickle(dev_filename)
        dev_data = dev_data.reset_index(drop=True)
        semeval_data = pd.read_pickle(semeval_filename)
        semeval_data = semeval_data.reset_index(drop=True)

        # we keep only semeval data when training the classifier
        train_dataset, training_ids = get_dataset_from_dataframe(training_data,
                                                      tokenizer=tokenizer,
                                                      args=args,
                                                      max_length=args.max_seq_length,
                                                      semeval_only=True)

        # validation with mednli and multinli
        validation_dataset, _ = get_dataset_from_dataframe(dev_data,
                                                           tokenizer=tokenizer,
                                                           args=args,
                                                           max_length=args.max_seq_length)

        # validation with semeval data only (not shuffled)
        semeval_dataset, all_ids = get_dataset_from_dataframe(semeval_data,
                                                              tokenizer=tokenizer,
                                                              args=args,
                                                              max_length=args.max_seq_length)

    elif args.dataset == 'DATASET_ONE':

        semeval_datafolder = os.path.join(args.data_folder, 'preprocessed', args.dataset)
        train_filename = os.path.join(semeval_datafolder, 'dataset_1_mnli_mednli_semeval.pkl')
        dev_filename = os.path.join(semeval_datafolder, 'dataset_1_dev_semeval23.pkl')

        training_data = pd.read_pickle(train_filename)
        training_data = training_data.reset_index(drop=True)
        dev_data = pd.read_pickle(dev_filename)
        dev_data = dev_data.reset_index(drop=True)

        train_dataset, _ = get_dataset_from_dataframe(training_data,
                                                   tokenizer=tokenizer,
                                                   args=args,
                                                   max_length=args.max_seq_length)

        validation_dataset, _ = get_dataset_from_dataframe(dev_data,
                                                        tokenizer=tokenizer,
                                                        args=args,
                                                        max_length=args.max_seq_length)

    elif args.dataset == 'SEMEVAL23_RAW':
        train_filename = os.path.join(args.data_folder, 'training_data', "train.json")
        dev_filename = os.path.join(args.data_folder, 'training_data', "dev.json")

        train_file = open(train_filename, 'r')
        train_data = json.load(train_file)
        train_file.close()
        dev_file = open(dev_filename, 'r')
        dev_data = json.load(dev_file)
        dev_file.close()

        train_dataset, _, _ = get_balanced_dataset_two_labels(train_data,
                                                        tokenizer=tokenizer,
                                                        max_length=args.max_seq_length)
        validation_dataset, _, _ = get_balanced_dataset_two_labels(dev_data,
                                                             tokenizer=tokenizer,
                                                             max_length=args.max_seq_length)
    elif args.dataset == 'SEMEVAL23':
        semeval_datafolder = os.path.join(args.data_folder, 'preprocessed', 'SEMEVAL23')
        train_filename = os.path.join(semeval_datafolder, 'balanced_training_dataset.pkl')
        dev_filename = os.path.join(semeval_datafolder, 'balanced_dev_dataset.pkl')

        training_data = pd.read_pickle(train_filename)
        training_data = training_data.reset_index(drop=True)

        dev_data = pd.read_pickle(dev_filename)
        dev_data = dev_data.reset_index(drop=True)

        train_dataset = get_balanced_dataset_three_labels(training_data,
                                             tokenizer=tokenizer,
                                             max_length=args.max_seq_length)

        validation_dataset = get_balanced_dataset_three_labels(dev_data,
                                                  tokenizer=tokenizer,
                                                  max_length=args.max_seq_length)

    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

        shuffle = True if args.shuffle else False
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                  num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        validate_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                     num_workers=args.workers, pin_memory=True, sampler=validation_sampler)

    stopper = EarlyStopping(min_delta=1e-5)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            validation_sampler.set_epoch(epoch)

        time1 = time.time()
        loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.7f}, accuracy {:.2f}'
              .format(epoch, time2 - time1, loss, train_acc))

        v_time1 = time.time()
        validation_loss, semeval_loss, acc = validate(validate_loader, semeval_dataset, all_ids, model, classifier,
                                                      criterion, epoch, args)
        v_time2 = time.time()
        print('epoch {}, total time {:.2f}, validation loss {:.7f}, semeval loss {:.7f}, validation accuracy {:.2f}'
              .format(epoch, v_time2 - v_time1, validation_loss, semeval_loss, acc))

        stopper(loss, semeval_loss)
        if stopper.early_stop:
            print("Early stop")
            break

        if best_acc1 is None or acc > best_acc1:
            best_acc1 = acc
            print('best accuracy: {:.3f}'.format(best_acc1.item()))
            save_file = os.path.join(args.save_folder, 'classifier_last.pth')
            save_model(classifier, optimizer, args, epoch, save_file, True)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        save_file = os.path.join(args.save_folder, 'classifier_last.pth')
        save_model(classifier, optimizer, args, epoch, save_file, False)
    print("best accuracy: {:.3f}".format(best_acc1.item()))


def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    top = AverageMeter('Accuracy', ':1.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.log_path, 'training_' + args.model_name + '.csv'))

    l1_criterion = nn.L1Loss(reduction='mean')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=args.eta)

    # switch to train mode
    model.eval()
    classifier.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)

        if args.gpu is not None:
            for i in range(1, len(batch)):
                batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

        # compute loss
        batch = tuple(t.cuda() for t in batch)
        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1],
                  "token_type_ids": batch[2]}
        with torch.no_grad():
            features = model(**inputs)

        logits = classifier(features.detach())
        labels = batch[3]
        loss = criterion(logits.view(-1, 2), labels.view(-1))

        # L1 regularization
        for param in model.encoder.parameters():
            loss += args.coefficient * l1_criterion(param, torch.zeros_like(param))

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps  # normalizes loss to account for batch accumulation

        losses.update(loss.item(), bsz)
        loss.backward()

        # update metric
        acc1 = accuracy(logits, labels)
        top.update(acc1[0].item(), bsz)

        learning.update(get_lr(optimizer), 1)

        # SGD
        if ((idx + 1) % args.gradient_accumulation_steps) == 0 or (idx + 1) == len(train_loader):
            optimizer.step()
            scheduler.step((epoch + idx) / len(train_loader))
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.log_epochs:
            progress.log_metrics(idx)

        # print info
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)
    return losses.avg, top.avg


def validate(val_loader, semeval_dataset, semeval_ids, model, classifier, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    top = AverageMeter('Accuracy', ':1.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.log_path, 'validation_' + args.model_name + '.csv'))

    semeval_batch_time = AverageMeter('Time', ':6.3f')
    semeval_losses = AverageMeter('Loss', ':1.5f')
    semeval_progress = ProgressMeter(
        len(semeval_ids),
        [semeval_batch_time, semeval_losses],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.log_path, 'semeval_validation_' + args.model_name + '.csv'))

    # switch to validate mode
    model.eval()
    classifier.eval()
    res = {"iid": [], "predicted": [], "gold_label": []}
    with torch.no_grad():
        end = time.time()
        # sem eval validation
        for i, _id in enumerate(semeval_ids):

            batch = [
                semeval_dataset[i][0].cuda(non_blocking=True),
                semeval_dataset[i][1].cuda(non_blocking=True),
                semeval_dataset[i][2].cuda(non_blocking=True),
                semeval_dataset[i][3].cuda(non_blocking=True)
            ]

            inputs = {"input_ids": batch[0].unsqueeze(0),
                      "attention_mask": batch[1].unsqueeze(0),
                      "token_type_ids": batch[2].unsqueeze(0)}

            features = model(**inputs)
            logits = classifier(features.detach())

            loss = criterion(logits.view(-1, 2), batch[3].view(-1))

            # normalizes loss to account for batch accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            semeval_losses.update(loss.item(), 1)

            _, _pred = logits.topk(1, 1, True, True)
            res["predicted"].append(_pred.item())
            res["iid"].append(_id)
            res["gold_label"].append(batch[3].item())

            # measure elapsed time
            semeval_batch_time.update(time.time() - end)
            end = time.time()

            if args.log_epochs:
                semeval_progress.log_metrics(i)

            # print info
            if (i + 1) % args.print_freq == 0:
                semeval_progress.display(i)

        results_df = pd.DataFrame(res)
        results_df = results_df.groupby('iid').aggregate(list).reset_index()
        results_df['_predicted'] = [1 if np.sum(row.predicted) > 0 else 0 for i, row in results_df.iterrows()]
        results_df['_gold_label'] = [1 if np.sum(row.gold_label) > 0 else 0 for i, row in results_df.iterrows()]

        acc = accuracy_score(results_df['_gold_label'], results_df['_predicted'], normalize=True)
        print(f'Sem Eval Validation Accuracy: {acc:.3f}')
        acc = torch.tensor([float(acc*100)]).cuda()

        for idx, batch in enumerate(val_loader):
            bsz = batch[0].size(0)

            if args.gpu is not None:
                for i in range(1, len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            labels = batch[3]
            features = model(**inputs)
            logits = classifier(features.detach())
            loss = criterion(logits.view(-1, 2), labels.view(-1))

            # normalizes loss to account for batch accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # update metric
            # print(logits)
            losses.update(loss.item(), bsz)
            acc1 = accuracy(logits, labels)
            top.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.log_epochs:
                progress.log_metrics(idx)

            # print info
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return losses.avg, semeval_losses.avg, acc


if __name__ == '__main__':
    main()
