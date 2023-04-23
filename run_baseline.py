import os
import time
import torch
import warnings
from util import save_model, parse_option, get_dataloaders, compute_real_accuracy, generate_results_file

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier
from pipeline.biomed_roberta_baseline import train as train_roberta, validate as validate_roberta
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")
__MODEL_SLUG__ = 'biomed'
best_acc = None


def main_worker(gpu, ngpu_per_node, args):
    global best_acc
    best_results = None
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

    validation_iids = dataloader_struct['iids']['validation']
    validation_trials = dataloader_struct['trials']['validation']
    validation_orders = dataloader_struct['orders']['validation']

    for epoch in range(args.epochs):
        time1 = time.time()
        loss, train_acc = train_roberta(training_loader, classifier, criterion, optimizer, scheduler, epoch, args)
        time2 = time.time()
        print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), loss))

        val_time1 = time.time()
        validation_loss, val_acc, result = validate_roberta(validation_loader, classifier, criterion, epoch, args,
                                                            extra_info=(validation_iids, validation_trials,
                                                                        validation_orders))
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (val_time2 - val_time1),
                                                                           validation_loss))

        semeval_majority_accuracy, semeval_at_least_one_accuracy = compute_real_accuracy(result)
        semeval_accuracy = max(semeval_majority_accuracy, semeval_at_least_one_accuracy)
        semeval_accuracy = torch.tensor([semeval_accuracy], dtype=torch.float32)

        if best_acc is None or semeval_accuracy > best_acc:
            best_acc = semeval_accuracy
            best_results = result
            print("New best accuracy: {:.3f}".format(best_acc.item()))
            save_file = os.path.join(args.save_folder, 'classifier_best.pth')
            save_model(classifier, optimizer, args, epoch, best_acc, save_file)
        print("Best accuracy: {:.3f}".format(best_acc.item()))

    generate_results_file(best_results, args, prefixes=['dev_majority_', 'dev_at_least_one_'])


if __name__ == '__main__':
    __args = parse_option()
    if __args.model_name[0:len(__MODEL_SLUG__)] != __MODEL_SLUG__:
        raise ValueError('Model name must be biomed')

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
        torch.cuda.manual_seed_all(__args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
        main_worker(__args.gpu, ngpus_per_node, __args)
