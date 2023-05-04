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
    return {"iid": [],
            "predicted_label": [],
            "trial": [],
            "genre": [],
            "order_": [],
            "gold_label": [],
            "predicted_evidence": [],
            "gold_evidence_label": [],
            "logits": []}.copy()


def add_metrics(dataset_name, bash_size, batch_index, iid_list, predicted_labels, true_labels, res, logits=None,
                order_list=None, trial_list=None, genres_list=None, unlabeled=False, predicted_evidence=None,
                gold_evidence_label=None):
    """
    Add metrics to the res dictionary.

    Args:
        predicted_evidence:
        gold_evidence_label:
        genres_list:
        logits:
        dataset_name: str
        bash_size: int
        batch_index: int
        iid_list: list
        predicted_labels: tensor. shape: (bash_size, num_classes)
        true_labels: tensor. shape: (bash_size, )
        res: dict
        order_list: list
        trial_list: list
        unlabeled: bool

    Returns:
    """
    res["predicted_label"] += predicted_labels.argmax(1).cpu().numpy().tolist()
    if predicted_evidence is not None:
        res["predicted_evidence"] += predicted_evidence.argmax(1).cpu().numpy().tolist()

    if not unlabeled:
        res["gold_label"] += true_labels.cpu().numpy().tolist()
        if gold_evidence_label is not None:
            res["gold_evidence_label"] += gold_evidence_label.cpu().numpy().tolist()

    if logits is not None:
        res["logits"] += logits.cpu().numpy().tolist()

    offset = batch_index * bash_size
    res["iid"] += iid_list[offset:offset + bash_size]
    if dataset_name == "nli4ct":
        res["trial"] += trial_list[offset:offset + bash_size]
        res["order_"] += order_list[offset:offset + bash_size]
    elif dataset_name == 'multinli':
        res["genre"] += genres_list[offset:offset + bash_size]
