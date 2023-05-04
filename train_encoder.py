import sys
import os
import time
import warnings
import pickle

from transformers import RobertaTokenizer
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW

from util import save_model, parse_option, get_dataloaders, epoch_summary, compute_real_accuracy, generate_results_file
from pipeline.encoder_pipeline import train as train_encoder, validate as validate_encoder

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.pairscl_model import PairSupConBert, RoBERTaForCL
from models.losses import SupConLoss
from torch.nn import CrossEntropyLoss

warnings.filterwarnings("ignore")
__MODEL_SLUG__ = ['pairscl', 'evidencescl']
minimal_loss = None


def main_worker(gpu, args):
    global minimal_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model = PairSupConBert(RoBERTaForCL.from_pretrained(
        "allenai/biomed_roberta_base",  # Use the 12-layer Biomed Roberta models from allenai, with a cased vocab.
        num_labels=args.max_seq_length,
        output_attentions=False,  # Whether the models return attentions weights.
        output_hidden_states=False,  # Whether the models return all hidden-states.
    ))
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_scl = SupConLoss(temperature=args.temp).cuda(args.gpu)
    criterion_ce = CrossEntropyLoss().cuda(args.gpu)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # optionally resume from a checkpoint
    if args.resume:
        if args.encoder_ckpt is not None and os.path.isfile(args.encoder_ckpt):
            checkpoint = torch.load(args.encoder_ckpt)
            model.load_state_dict(checkpoint['models'])
        else:
            try:
                checkpoint = torch.load(os.path.join(args.save_folder, 'encoder.pth'),
                                        map_location=args.device)
                model.load_state_dict(checkpoint['models'])
            except FileNotFoundError:
                raise "No encoder checkpoint found. Please specify a checkpoint to load or ensure a encoder."

        args.start_epoch = checkpoint['epoch']
        args.epochs = args.start_epoch + args.epochs
        optimizer.load_state_dict(checkpoint['optimizer'])
        minimal_loss = checkpoint['best_metric']
        print("=> Epoch from the loaded encoder checkpoint: '{}' (epoch {})"
              .format(args.encoder_ckpt, checkpoint['epoch']))

    cudnn.benchmark = True

    # load data
    dataloader_struct = get_dataloaders(args.dataset, args.data_folder, tokenizer, args.batch_size, args.workers,
                                        args.max_seq_length, args.num_classes)

    training_loader = dataloader_struct['loader']['training']
    validation_loader = dataloader_struct['loader']['validation']

    train_iids = dataloader_struct['iids']['training']
    train_trials = dataloader_struct['trials']['training']
    train_orders = dataloader_struct['orders']['training']
    train_genres = dataloader_struct['genres']['training']

    val_iids = dataloader_struct['iids']['validation']
    val_trials = dataloader_struct['trials']['validation']
    val_orders = dataloader_struct['orders']['validation']
    val_genres = dataloader_struct['genres']['validation']

    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()

        train_loss, train_acc, train_res = train_encoder(training_loader, model, criterion_scl, criterion_ce, optimizer,
                                                         scheduler, epoch, args, extra_info=(train_iids,
                                                                                             train_trials,
                                                                                             train_orders,
                                                                                             train_genres))
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), train_loss))
        train_loss = torch.tensor([train_loss], dtype=torch.float32)
        train_semeval_maj_acc, train_semeval_at_least_one_acc, train_agg_results = compute_real_accuracy(train_res)
        train_semeval_acc = max(train_semeval_maj_acc, train_semeval_at_least_one_acc)

        # evaluate on validation set
        val_time1 = time.time()
        val_loss, val_acc, val_res = validate_encoder(validation_loader, model, criterion_scl, criterion_ce, epoch,
                                                      args, extra_info=(val_iids,
                                                                        val_trials,
                                                                        val_orders,
                                                                        val_genres))
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, validation loss {:.7f}'.format(epoch, (val_time2 - val_time1),
                                                                                      val_loss))
        val_loss = torch.tensor([val_loss], dtype=torch.float32)
        val_semeval_maj_acc, val_semeval_at_least_one_acc, val_agg_results = compute_real_accuracy(val_res)
        val_semeval_acc = max(val_semeval_maj_acc, val_semeval_at_least_one_acc)

        if minimal_loss is None or val_loss < minimal_loss:
            minimal_loss = val_loss
            best_val_results = val_agg_results
            print("New minimal loss: {:.5f}".format(minimal_loss.item()))
            save_file = os.path.join(args.save_folder, 'encoder_best.pth')
            save_model(model, optimizer, args, epoch, minimal_loss, save_file)

        # display epoch summary
        epoch_summary(args.model_name, epoch, train_loss.item(), val_loss.item(), minimal_loss.item(), 'loss')

    # save the last model
    save_file = os.path.join(args.save_folder, 'encoder_last.pth')
    save_model(model, optimizer, args, epoch, minimal_loss, save_file)

    # save results in semeval format
    generate_results_file(train_agg_results, args, prefixes=['train_majority_', 'train_at_least_one_'])
    generate_results_file(best_val_results, args, prefixes=['val_majority_', 'val_at_least_one_'])

    # save raw results in pickle
    with open(args.save_folder + '/train_results_' + args.model_name + '.pkl', 'wb') as f:
        pickle.dump(train_res, f)
    with open(args.save_folder + '/validation_results_' + args.model_name + '.pkl', 'wb') as f:
        pickle.dump(val_res, f)


if __name__ == '__main__':
    __args = parse_option()
    valid_model = [__args.model_name[0:len(__model_slug)] == __model_slug for __model_slug in __MODEL_SLUG__]
    if not any(valid_model):
        raise ValueError('Model name must be one of {}'.format(__MODEL_SLUG__))

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
        torch.cuda.manual_seed(__args.seed)
        torch.cuda.manual_seed_all(__args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if __args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if __args.dist_url == "env://" and __args.world_size == -1:
        __args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    __args.distributed = __args.world_size > 1 or __args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if __args.multiprocessing_distributed:
        __args.world_size = ngpus_per_node * __args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, __args,))
    else:
        main_worker(__args.gpu, __args)

    print('Done!')
    sys.exit(0)
