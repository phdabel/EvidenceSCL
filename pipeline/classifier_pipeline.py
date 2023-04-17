import os
import time

import torch
import torch.nn as nn

from ..util import AverageMeter, ProgressMeter, get_lr
from sklearn.metrics import accuracy_score


def train(dataloader, model, classifier, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    top = AverageMeter('Accuracy', ':1.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    l1_criterion = nn.L1Loss()

    # switch to train mode
    model.eval()
    classifier.train()

    end = time.time()
    for idx, batch in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)
        batch = tuple(t.cuda() for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}

        with torch.no_grad():
            features = model(**inputs)

        logits = classifier(features.detach())
        predicted_labels = logits.view(-1, args.num_classes)
        true_labels = inputs['labels'].view(-1)
        labels = inputs['labels']
        loss = criterion(predicted_labels, true_labels)

        # L1 regularization
        if args.l1_regularization > 0:
            for param in classifier.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        # update metrics
        losses.update(loss.item(), bsz)
        loss.backward()

        acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
        top.update(acc, bsz)
        learning.update(get_lr(optimizer), bsz)

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(dataloader)):
            optimizer.step()
            scheduler.step((epoch + idx) / len(dataloader))
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.log_metrics(idx)
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)

    return losses.avg, top.avg


def validate(dataloader, model, classifier, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    top = AverageMeter('Accuracy', ':1.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    classifier.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}

            features = model(**inputs)
            logits = classifier(features.detach())

            predicted_labels = logits.view(-1, args.num_classes)
            true_labels = inputs['labels'].view(-1)
            labels = inputs['labels']
            loss = criterion(predicted_labels, true_labels)

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            # update metrics
            losses.update(loss.item(), bsz)
            acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
            top.update(acc, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.log_metrics(idx)
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return losses.avg, top.avg
