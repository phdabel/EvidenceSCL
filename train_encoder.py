import os
import time
from transformers import RobertaTokenizer
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW

from util import save_model, parse_option, get_dataloaders
from pipeline.encoder_pipeline import train as train_encoder, validate as validate_encoder

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.pairscl_model import PairSupConBert, RoBERTaForCL
from models.losses import SupConLoss
from torch.nn import CrossEntropyLoss


def main_worker(args):
    
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
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        print("=> Loading checkpoint from: '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Epoch from the loaded checkpoint: '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.ckpt))

    cudnn.benchmark = True

    training_loader, validation_loader, test_loader, train_iids, validate_iids, test_iids = \
        get_dataloaders(args.dataset, args.data_folder, tokenizer, args.batch_size, args.workers,
                        args.max_seq_length, args.num_classes)

    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        loss = train_encoder(training_loader, model, criterion_scl, criterion_ce, optimizer, scheduler, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), loss))

        # evaluate on validation set
        val_time1 = time.time()
        validation_loss = validate_encoder(validation_loader, model, criterion_scl, criterion_ce, epoch, args)
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, validation loss {:.7f}'.format(epoch, (val_time2 - val_time1),
                                                                                      validation_loss))

        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            save_file = os.path.join(args.save_folder, 'encoder.pth')
            save_model(model, optimizer, args, epoch, None, save_file)


if __name__ == '__main__':
    __args = parse_option()
    main_worker(__args)
