import time
import torch
from .util import AverageMeter, ProgressMeter, add_metrics, create_metrics_dict, compute_metric

__NLI4CT_SLUG__ = "nli4ct"


def run_classifier_test(dataloader, classifier, args, extra=None):
    """

    Args:
        dataloader: torch.utils.data.DataLoader
        classifier: models.linear_classifier.LinearClassifier
        args: argparse.ArgumentParser
        extra: tuple of (iids, trials, orders, unlabeled) if args.dataset == "nli4ct"

    Returns:
    """
    iids, trials, sentence_orders, genre_list, unlabeled, types = extra
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter(args.evaluation_metric.capitalize(), ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, top])

    classifier.eval()
    res = create_metrics_dict()

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}

            output = classifier(**inputs)
            logits = output[0].view(-1, args.num_classes)
            _, prediction = logits.topk(1, 1, True, True)

            predictions = logits
            true_nli_labels = None if unlabeled else batch[3].view(-1)
            true_er_labels = None if unlabeled else batch[4].view(-1)
            true_labels = None if unlabeled else true_nli_labels if args.task == 'nli' else true_er_labels

            # add metrics to res dictionary
            evaluate_dataset = args.evaluate_dataset if args.evaluate_dataset is not None else args.dataset

            add_metrics(dataset_name=evaluate_dataset, bash_size=bsz, batch_index=idx, iid_list=iids,
                        predicted_labels=predictions if args.task == 'nli' else None,
                        true_labels=true_nli_labels,
                        res=res, logits=logits, order_list=sentence_orders, trial_list=trials, itype_list=types,
                        genres_list=genre_list, unlabeled=unlabeled,
                        predicted_evidence=predictions if args.task == 'ir' else None,
                        gold_evidence_label=true_er_labels)

            if extra is not None and not unlabeled:
                metric = compute_metric(true_labels=true_labels.cpu().numpy(),
                                        predicted_labels=predictions.argmax(1).cpu().numpy(),
                                        args=args)

                top.update(metric, bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return (res, top.avg) if not unlabeled else (res, None)
