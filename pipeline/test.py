import time
import pickle
import torch
from .util import AverageMeter
from sklearn.metrics import accuracy_score

__NLI4CT_SLUG__ = "nli4ct"


def run_test(dataloader, classifier, args, extra=None):
    """

    Args:
        dataloader: torch.utils.data.DataLoader
        classifier: models.linear_classifier.LinearClassifier
        args: argparse.ArgumentParser
        extra: tuple of (iids, trials, orders, unlabeled) if args.dataset == "nli4ct"

    Returns:
    """
    iids, trials, sentence_orders, unlabeled = extra
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Accuracy', ':1.3f')

    classifier.eval()

    res = {"iid": [], "predicted_label": [], "trial": [], "order_": [], "gold_label": [], "predicted_evidence": [],
           "gold_evidence_label": [], "logits": []}

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}

            logits = classifier(inputs)
            _, prediction = logits.topk(1, 1, True, True)
            res["predicted_label"] += prediction.view(-1, args.num_classes).cpu().numpy().tolist()
            res["logits"] += logits.cpu().numpy().tolist()

            if extra is not None and args.dataset == __NLI4CT_SLUG__:
                offset = idx * bsz
                res["iid"] += iids[offset:offset + bsz]
                res["trial"] += trials[offset:offset + bsz]
                res["order_"] += sentence_orders[offset:offset + bsz]

            if extra is not None and not unlabeled:
                res["gold_label"] += batch[3].view(-1).cpu().numpy().tolist()
                acc = accuracy_score(res["gold_label"], res["predicted_label"])
                top.update(acc, bsz)

            batch_time.update(time.time() - end)
            end = time.time()

    # save results in pickle file
    with open(args.save_folder + '/test_results_' + args.model_name + '.pkl', 'wb') as f:
        pickle.dump(res, f)

    return top.avg if not unlabeled else None
