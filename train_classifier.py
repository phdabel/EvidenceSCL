import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from transformers import RobertaTokenizer
from models.pairscl_model import RoBERTaForCL, PairSupConBert, LinearClassifier
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pipeline.classifier_pipeline import train as train_classifier, validate as validate_classifier

from util import parse_option, get_dataloaders, save_model


def main_worker(args):
    best_acc = None
    model = PairSupConBert(RoBERTaForCL.from_pretrained(
        "allenai/biomed_roberta_base",  # Use the 12-layer Biomed Roberta models from allenai, with a cased vocab.
        num_labels=args.max_seq_length,
        output_attentions=False,  # Whether the models return attentions weights.
        output_hidden_states=False,  # Whether the models return all hidden-states.
    ))
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")

    classifier = LinearClassifier(RoBERTaForCL.from_pretrained(
        "allenai/biomed_roberta_base",  # Use the 12-layer Biomed Roberta models from allenai, with a cased vocab.
        num_labels=args.max_seq_length,
        output_attentions=False,  # Whether the models return attentions weights.
        output_hidden_states=False,  # Whether the models return all hidden-states.
    ), num_classes=args.num_classes)

    encoder_ckpt = torch.load(args.encoder_ckpt)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    criterion_ce = CrossEntropyLoss().cuda(args.gpu)
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    model.load_state_dict(encoder_ckpt['models'])

    # optionally resume from a checkpoint
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        print("=> Loading checkpoint from: '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        args.start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Epoch from the loaded checkpoint: '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(args.ckpt))

    cudnn.benchmark = True

    training_loader, validation_loader, test_loader, train_iids, validate_iids, test_iids = \
        get_dataloaders(args.dataset, args.data_folder, tokenizer, args.batch_size, args.workers,
                        args.max_seq_length, args.num_classes)

    for epoch in range(args.start_epoch, args.epochs):

        time1 = time.time()
        loss, train_acc = train_classifier(training_loader, model, classifier, criterion_ce, optimizer, scheduler,
                                           epoch, args)
        time2 = time.time()
        print('Epoch {}, total time {:.2f}, training accuracy {:.3f}'.format(epoch, time2 - time1, train_acc))

        val_time1 = time.time()
        val_loss, val_acc = validate_classifier(validation_loader, model, classifier, criterion_ce, epoch, args)
        val_time2 = time.time()
        print('Epoch {}, total time {:.2f}, validation accuracy {:.3f}'.format(epoch, val_time2 - val_time1, val_acc))

        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            print("New best accuracy: {:.3f}".format(best_acc.item()))
            save_file = os.path.join(args.save_folder, 'classifier_best.pth')
            save_model(classifier, optimizer, args, epoch, best_acc, save_file)
        print("Best accuracy: {:.3f}".format(best_acc.item()))


if __name__ == '__main__':
    __args = parse_option()

    if __args.encoder_ckpt is None or not os.path.isfile(__args.encoder_ckpt):
        raise ValueError("=> No checkpoint found at '{}'".format(__args.encoder_ckpt))

    main_worker(__args)
