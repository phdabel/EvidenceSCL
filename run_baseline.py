import os
import random
import time
import warnings
import torch
import torch.backends.cudnn as cudnn

from util import save_model, parse_option, get_dataloaders

from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier

from pipeline.biomed_roberta_baseline import train as train_roberta, validate as validate_roberta
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


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

    training_loader, validation_loader, test_loader, train_iids, validate_iids, test_iids = \
        get_dataloaders(args.dataset, args.data_folder, tokenizer, args.batch_size, args.workers,
                        args.max_seq_length, args.num_classes)

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
