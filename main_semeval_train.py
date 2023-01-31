import os
import argparse
import random
import time
import math
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from preprocessing.semeval_dataset import get_balanced_dataset_three_labels, get_balanced_dataset_two_labels, \
    get_dataset_from_dataframe
from util import adjust_learning_rate, warmup_learning_rate, save_model, \
    AverageMeter, ProgressMeter, NLIProcessor, load_and_cache_examples, EarlyStopping
from torch.utils.data import DataLoader
from bert_model import PairSupConBert, BertForCL
from losses import SupConLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # model dataset
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--model', type=str, default='ROBERTA')
    parser.add_argument('--dataset', type=str, default='SEMEVAL23',
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets', help='path to custom dataset')
    # training
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,8',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.01,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')  # weight decay corresponds to L2 regularization factor
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64,
                        help='number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    # distribute
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=42, type=int,
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
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="the parameter to balance the training objective (default: 1.0)")
    parser.add_argument('--coefficient', type=float, default=0.001,
                        help='L1 regularization coefficient')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function (default: 0.05)')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--eta', type=float, default=1e-5,
                        help='minimum value')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    args = parser.parse_args()
    args.model_path = './save/{}_models'.format(args.dataset)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}'.\
        format(args.dataset, args.model, args.learning_rate,
               args.weight_decay, args.batch_size, args.temp)

    if args.cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    # warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def validate(validation_loader, model, criterion_sup, criterion_ce, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.7f')
    progress = ProgressMeter(
        len(validation_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(validation_loader):
            bsz = batch[0].size(0)

            if args.gpu is not None:
                for i in range(1, len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            feature1, feature2 = model(**inputs)

            loss_sup = criterion_sup(feature2, batch[3])
            loss_ce = criterion_ce(feature1, batch[3])
            loss = loss_sup + args.alpha * loss_ce

            # normalizes loss to account for batch accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # update metric
            # print(logits)
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)
    return losses.avg


def train(train_loader, model, criterion_sup, criterion_ce, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.7f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    l1_criterion = nn.L1Loss(reduction='mean')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-06)

    # switch to train mode
    model.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)

        if args.gpu is not None:
            for i in range(1, len(batch)):
                batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

        # warm-up learning rate
        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        batch = tuple(t.cuda() for t in batch)
        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1],
                  "token_type_ids": batch[2]}
        feature1, feature2 = model(**inputs)

        loss_sup = criterion_sup(feature2, batch[3])
        loss_ce = criterion_ce(feature1, batch[3])
        loss = loss_sup + args.alpha * loss_ce

        # L1 regularization
        for param in model.parameters():
            loss += args.coefficient * l1_criterion(param, torch.zeros_like(param))

        # normalizes loss to account for batch accumulation
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # update metrics
        losses.update(loss.item(), bsz)

        # AdamW
        loss.backward()

        if ((idx + 1) % args.gradient_accumulation_steps) == 0 or (idx + 1) == len(train_loader):
            optimizer.step()
            scheduler.step((epoch + idx)/len(train_loader))
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)
    return losses.avg


def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
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
    global best_acc1
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
        num_labels=args.max_seq_length,  # max sequence length (128)
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ), num_classes=2)  # number of classes (0 - contradiction, 1 - entailment)

    tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_supcon = SupConLoss(temperature=args.temp).cuda(args.gpu) 
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
            print("Best ACC %.2f", best_acc1)
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

    validation_dataset = None
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
        # dev_filename = os.path.join(args.data_folder, 'training_data', "dev.json")

        train_file = open(train_filename, 'r')
        train_data = json.load(train_file)
        train_file.close()

        train_dataset, _, _ = get_balanced_dataset_two_labels(train_data,
                                                              tokenizer=tokenizer,
                                                              max_length=args.max_seq_length)

    elif args.dataset == 'SEMEVAL23':
        semeval_datafolder = os.path.join(args.data_folder, 'preprocessed', 'SEMEVAL23')
        train_filename = os.path.join(semeval_datafolder, 'balanced_training_dataset.pkl')

        training_data = pd.read_pickle(train_filename)
        training_data = training_data.reset_index(drop=True)

        train_dataset = get_balanced_dataset_three_labels(training_data,
                                                          tokenizer=tokenizer,
                                                          max_length=args.max_seq_length)

    elif args.dataset == 'SNLI' or args.dataset == "MNLI":
        train_file = os.path.join(args.data_folder, "preprocessed", args.dataset, "train_data.pkl")

        print("load dataset")
        with open(train_file, "rb") as pkl:
            processor = NLIProcessor(pickle.load(pkl))

        train_dataset = load_and_cache_examples(args, processor, tokenizer, "train", args.dataset)

    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        validation_sampler = None
        if validation_dataset is not None:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

    shuffle = True if args.shuffle else False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                              num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                       num_workers=args.workers, pin_memory=True, sampler=validation_sampler)



    # handle - save each epoch
    # model_save_file = os.path.join(args.model_path, f'{args.model_name}.pt')
    # epoch_save_file = os.path.join(args.model_path, f'{args.model_name}_epoch_data.pt')
    stopper = EarlyStopping()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if validation_dataset is not None:
                validation_sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)

        time1 = time.time()
        loss = train(train_loader, model, criterion_supcon, criterion_ce, optimizer, epoch, args)
        time2 = time.time()
        print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), loss))

        val_time1 = time.time()
        validation_loss = validate(validation_loader, model, criterion_supcon, criterion_ce, epoch, args)
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (val_time2 - val_time1), validation_loss))

        stopper(loss, validation_loss)
        if stopper.early_stop:
            print("Early stop")
            break
        
    # save the last model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        save_file = os.path.join(args.save_folder, 'last.pth')
        save_model(model, optimizer, args, args.epochs, save_file, False)  
 

if __name__ == '__main__':
    main()
