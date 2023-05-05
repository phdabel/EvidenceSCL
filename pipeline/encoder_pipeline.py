import os
import time
from pipeline.util import AverageMeter, ProgressMeter, get_lr

import torch
import torch.nn as nn
from .util import add_metrics, create_metrics_dict
from sklearn.metrics import accuracy_score


def train(dataloader, model, criterion_scl, criterion_ce, optimizer, scheduler, epoch, args, extra_info=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    top = AverageMeter('Accuracy', ':1.5f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses, top],
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
                  'token_type_ids': batch[2]}

        feature1, feature2 = model(**inputs)

        true_labels = batch[3].view(-1) if not args.evidence_retrieval else batch[4].view(-1)
        true_evidence_labels = torch.tensor(true_labels < 2, dtype=torch.long) \
            if 'evidencescl' in args.model_name and args.num_classes == 3 else true_labels
        predicted_labels = feature1.view(-1, args.num_classes)

        # add metrics to res dictionary
        add_metrics(dataset_name=args.dataset, bash_size=bsz, batch_index=idx, iid_list=iid_list,
                    predicted_labels=predicted_labels, true_labels=true_labels, res=res, logits=None,
                    order_list=order_list, trial_list=trial_list, genres_list=genre_list, unlabeled=False,
                    predicted_evidence=None, gold_evidence_label=true_evidence_labels)

        loss_scl = criterion_scl(feature2, true_evidence_labels)
        loss_ce = criterion_ce(predicted_labels, true_labels)
        loss = loss_scl + loss_ce * args.alpha

        # L1 regularization
        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        # gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
        # update metrics
        top.update(acc, bsz)
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

    return losses.avg, top.avg, res


def validate(dataloader, model, criterion_scl, criterion_ce, epoch, args, extra_info=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    top = AverageMeter('Accuracy', ':1.5f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top],
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
                      'token_type_ids': batch[2]}

            feature1, feature2 = model(**inputs)
            true_labels = batch[3].view(-1) if not args.evidence_retrieval else batch[4].view(-1)
            true_evidence_labels = torch.tensor(true_labels < 2, dtype=torch.long) \
                if 'evidencescl' in args.model_name and args.num_classes == 3 else true_labels
            predicted_labels = feature1.view(-1, args.num_classes)

            # add metrics to res dictionary
            add_metrics(dataset_name=args.dataset, bash_size=bsz, batch_index=idx, iid_list=iid_list,
                        predicted_labels=predicted_labels if not args.evidence_retrieval else None,
                        true_labels=true_labels, res=res, logits=None, order_list=order_list, trial_list=trial_list,
                        genres_list=genre_list, unlabeled=False,
                        predicted_evidence=predicted_labels if args.evidence_retrieval else None,
                        gold_evidence_label=true_evidence_labels)

            loss_scl = criterion_scl(feature2, true_evidence_labels)
            loss_ce = criterion_ce(predicted_labels, true_labels)
            loss = loss_scl + loss_ce * args.alpha

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            # update metric
            losses.update(loss.item(), bsz)
            acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.argmax(1).cpu().numpy())
            top.update(acc, bsz)

            # measuring elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.log_metrics(idx)
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return losses.avg, top.avg, res
