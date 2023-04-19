import os
import time
from pipeline.util import AverageMeter, ProgressMeter, get_lr
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn


def train(dataloader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    top = AverageMeter('Accuracy', ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    l1_criterion = nn.L1Loss()
    model.train()
    end = time.time()
    for idx, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[3]}

        output = model(**inputs)
        predicted_labels = output[1].view(-1, args.num_classes)
        true_labels = inputs['labels'].view(-1)
        loss = criterion(predicted_labels, true_labels)

        # L1 regularization
        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        # gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        losses.update(loss.item(), bsz)
        loss.backward()
        # update metrics
        acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
        top.update(acc, bsz)
        learning.update(get_lr(optimizer), bsz)

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

    return losses.avg, top.avg


def validate(dataloader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    top = AverageMeter('Accuracy', ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            output = model(**inputs)
            predicted_labels = output[1].view(-1, args.num_classes)
            true_labels = batch[3].view(-1)

            loss = criterion(predicted_labels, true_labels)

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            losses.update(loss.item(), bsz)
            acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
            top.update(acc, bsz)

            batch_time.update(time.time() - end)
            end = time.time()
            progress.log_metrics(idx)

            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return losses.avg, top.avg
