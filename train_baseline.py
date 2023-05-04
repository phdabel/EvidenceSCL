import sys
import os
import time
import torch
import warnings
from util import save_model, parse_option, get_dataloaders, compute_real_accuracy, generate_results_file, epoch_summary

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaModel
import torch.backends.cudnn as cudnn
from models.linear_classifier import LinearClassifier
from pipeline.biomed_roberta_baseline import train as train_roberta, validate as validate_roberta
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")
__MODEL_SLUG__ = 'biomed'
best_val_acc = None


def main_worker(gpu, args):
    global best_val_acc
    best_val_results = None
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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

    epoch, val_semeval_acc, train_agg_results = None, None, None
    for epoch in range(args.epochs):
        time1 = time.time()
        train_loss, train_acc, train_result = train_roberta(training_loader, classifier, criterion, optimizer,
                                                            scheduler, epoch, args,
                                                            extra_info=(train_iids,
                                                                        train_trials,
                                                                        train_orders,
                                                                        train_genres))
        time2 = time.time()
        print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), train_loss))
        train_semeval_maj_acc, train_semeval_at_least_one_acc, train_agg_results = compute_real_accuracy(train_result)
        train_semeval_acc = max(train_semeval_maj_acc, train_semeval_at_least_one_acc)
        train_semeval_acc = torch.tensor([train_semeval_acc], dtype=torch.float32)

        val_time1 = time.time()
        val_loss, val_acc, val_result = validate_roberta(validation_loader, classifier, criterion, epoch, args,
                                                         extra_info=(val_iids,
                                                                     val_trials,
                                                                     val_orders,
                                                                     val_genres))
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (val_time2 - val_time1),
                                                                           val_loss))

        val_semeval_maj_acc, val_semeval_at_least_one_acc, val_agg_results = compute_real_accuracy(val_result)
        val_semeval_acc = max(val_semeval_maj_acc, val_semeval_at_least_one_acc)
        val_semeval_acc = torch.tensor([val_semeval_acc], dtype=torch.float32)

        if best_val_acc is None or val_semeval_acc > best_val_acc:
            best_val_acc = val_semeval_acc
            best_val_results = val_agg_results
            print("New best accuracy: {:.3f}".format(best_val_acc.item()))
            save_file = os.path.join(args.save_folder, 'classifier_best.pth')
            save_model(classifier, optimizer, args, epoch, best_val_acc, save_file)

        # display epoch summary
        epoch_summary(args.model_name, epoch, train_semeval_acc.item(), val_semeval_acc.item(), best_val_acc.item())

    # save the last model
    save_file = os.path.join(args.save_folder, 'classifier_last.pth')
    save_model(classifier, optimizer, args, epoch, val_semeval_acc, save_file)

    generate_results_file(train_agg_results, args, prefixes=['train_majority_', 'train_at_least_one_'])
    generate_results_file(best_val_results, args, prefixes=['val_majority_', 'val_at_least_one_'])


if __name__ == '__main__':
    __args = parse_option()
    if __args.model_name[0:len(__MODEL_SLUG__)] != __MODEL_SLUG__:
        raise ValueError('Model name must be biomed')

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, __args))
    else:
        main_worker(__args.gpu, __args)

    print('Done!')
    sys.exit(0)
