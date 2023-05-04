import os
import time
from pipeline.util import AverageMeter, ProgressMeter, get_lr

import torch
import torch.nn as nn
from .util import add_metrics, create_metrics_dict


def train(dataloader, model, criterion_scl, criterion_ce, optimizer, scheduler, epoch, args, extra_info=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_encoder_' + args.model_name + '.csv'))

    res = create_metrics_dict()

    iid_list, trial_list, order_list, genre_list = None, None, None, None
    if extra_info is not None:
        iid_list, trial_list, order_list, genre_list = extra_info

    l1_criterion = nn.L1Loss()
    model.train()
    end = time.time()
    for idx, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)
        batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3]}

        feature1, feature2 = model(**inputs)
        scl_labels = inputs['labels']
        if 'evidencescl' in args.model_name and args.num_classes == 3:
            scl_labels = torch.tensor(scl_labels < 2, dtype=torch.long)

        loss_scl = criterion_scl(feature2, scl_labels)
        loss_ce = criterion_ce(feature1, inputs['labels'])
        loss = loss_scl + loss_ce * args.alpha

        # L1 regularization
        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        # gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        # update metrics
        losses.update(loss.item(), bsz)
        learning.update(get_lr(optimizer), bsz)
        loss.backward()

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(dataloader)):
            optimizer.step()
            scheduler.step((epoch + idx) / len(dataloader))
            optimizer.zero_grad()

        # measuring elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.log_metrics(idx)
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)

    return losses.avg, res


def validate(dataloader, model, criterion_scl, criterion_ce, epoch, args, extra_info=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_encoder_' + args.model_name + '.csv'))

    res = create_metrics_dict()

    iid_list, trial_list, order_list, genre_list = None, None, None, None
    if extra_info is not None:
        iid_list, trial_list, order_list, genre_list = extra_info

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            feature1, feature2 = model(**inputs)
            scl_labels = inputs['labels']
            if 'evidencescl' in args.model_name and args.num_classes == 3:
                scl_labels = torch.tensor(scl_labels < 2, dtype=torch.long)

            loss_scl = criterion_scl(feature2, scl_labels)
            loss_ce = criterion_ce(feature1, inputs['labels'])
            loss = loss_scl + loss_ce * args.alpha

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            # update metric
            losses.update(loss.item(), bsz)

            # measuring elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.log_metrics(idx)
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return losses.avg, res
