from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.short_name = name.lower().replace(' ', '_')
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __csv__(self):
        sep = ';'
        fmtstr = sep.join(['{short_name}',
                           '{val' + self.fmt + '}'])
        return fmtstr.format(**self.__dict__).split(sep)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logfile=""):
        self.batch_fmtstr = ProgressMeter._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile
        self.header = False

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def log_metrics(self, batch, sep=";"):
        epoch = int(self.prefix[8:-1])
        entries = dict()
        entries['epoch'] = epoch
        entries['batch'] = batch
        for i, meter in enumerate(self.meters):
            metric, value = meter.__csv__()
            entries[metric] = float(value)

        _logfile = open(self.logfile, 'a')
        if not self.header and epoch == 0:
            self.header = sep.join([key for key in entries.keys()]) + '\n'
            _logfile.write(self.header)
            self.header = True

        content = sep.join([str(entries[key]) for key in entries.keys()]) + '\n'
        _logfile.write(content)
        _logfile.close()

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def create_metrics_dict():
    return {"iid": [],                  # instance id
            "predicted_label": [],      # predicted label
            "review": [],               # review id           (robin dataset)
            "study": [],                # study id            (robin dataset)
            "type": [],                 # type id             (robin dataset)
            "trial": [],                # trial id            (nli4ct dataset)
            "itype": [],                # instance type       (nli4ct dataset)
            "genre": [],                # genre               (multinli dataset)
            "order_": [],               # order               (nli4ct dataset)
            "gold_label": [],           # gold label
            "predicted_evidence": [],   # predicted evidence  (nli4ct dataset)
            "gold_evidence_label": [],  # gold evidence label (nli4ct dataset)
            "logits": []                # logits
            }.copy()


def add_metrics(dataset_name, batch_size, batch_index, iid_list, predicted_labels, true_labels, res, logits=None,
                order_list=None, trial_list=None, itype_list=None, genres_list=None, unlabeled=False,
                predicted_evidence=None, gold_evidence_label=None, review_list=None, study_list=None, type_list=None):
    """
    Add metrics to the res dictionary.

    Args:
        predicted_evidence:
        gold_evidence_label:
        genres_list:
        logits:
        dataset_name: str
        batch_size: int
        batch_index: int
        iid_list: list
        predicted_labels: tensor. shape: (batch_size, num_classes)
        true_labels: tensor.shape: (batch_size, )
        res: dict
        order_list: list
        trial_list: list
        itype_list: list
        unlabeled: bool

    Returns:
    """
    if predicted_labels is not None:
        res["predicted_label"] += predicted_labels.argmax(1).cpu().numpy().tolist()
    if predicted_evidence is not None:
        res["predicted_evidence"] += predicted_evidence.argmax(1).cpu().numpy().tolist()

    if not unlabeled:
        res["gold_label"] += true_labels.cpu().numpy().tolist()
        if gold_evidence_label is not None:
            res["gold_evidence_label"] += gold_evidence_label.cpu().numpy().tolist()

    if logits is not None:
        res["logits"] += logits.cpu().numpy().tolist()

    offset = batch_index * batch_size

    if dataset_name == "nli4ct":
        res["trial"] += trial_list[offset:offset + batch_size]
        res["order_"] += order_list[offset:offset + batch_size]
        res["itype"] += itype_list[offset:offset + batch_size]
    elif dataset_name == 'multinli':
        res["genre"] += genres_list[offset:offset + batch_size]
    elif dataset_name == 'robin':
        res["review"] += review_list[offset:offset + batch_size]
        res["study"] += study_list[offset:offset + batch_size]
        res["type"] += type_list[offset:offset + batch_size]

    if dataset_name == 'robin':
        res["iid"] += iid_list.cpu().numpy().tolist()
    else:
        res["iid"] += iid_list[offset:offset + batch_size]


def compute_metric(true_labels, predicted_labels, args):
    _average = 'binary' if args.num_classes == 2 else 'macro'
    if args.evaluation_metric == 'f1':
        return f1_score(true_labels, predicted_labels, average=_average)
    elif args.evaluation_metric == 'precision':
        return precision_score(true_labels, predicted_labels, average=_average)
    elif args.evaluation_metric == 'recall':
        return recall_score(true_labels, predicted_labels, average=_average)

    return accuracy_score(true_labels, predicted_labels)
