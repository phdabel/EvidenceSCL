import os
import torch
import warnings
from util import parse_option, get_dataloader, generate_results_file

import torch.backends.cudnn as cudnn
from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier
from pipeline.test import run_test as test_biomed_roberta

warnings.filterwarnings("ignore")
__MODEL_SLUG__ = 'biomed'


def main_worker(args):
    classifier = LinearClassifier(RobertaModel.from_pretrained("allenai/biomed_roberta_base"),
                                  num_classes=args.num_classes)
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        classifier = torch.nn.DataParallel(classifier).cuda()

    if args.ckpt is not None:
        classifier_ckpt = torch.load(args.ckpt, map_location=args.device)
        # classifier_state_dict = {key[7:]: classifier_ckpt['models'][key] for key in classifier_ckpt['models'].keys()}
        classifier.load_state_dict(classifier_ckpt["models"])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.ckpt, classifier_ckpt['epoch']))
    else:
        try:
            classifier_ckpt = torch.load(os.path.join(args.save_folder, 'classifier_best.pth'),
                                         map_location=args.device)
            classifier.load_state_dict(classifier_ckpt["models"])
            print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(args.save_folder, 'classifier_best.pth'),
                                                                classifier_ckpt['epoch']))
        except FileNotFoundError:
            raise "No classifier checkpoint found. Please specify a checkpoint to load or ensure a classifier"

    cudnn.benchmark = True

    # load test data
    test_loader, iids, trials, orders = get_dataloader(args.data_folder, args.dataset, "nli4ct_unlabeled_test.pkl",
                                                       tokenizer, args.batch_size, args.workers,
                                                       args.max_seq_length, test=True)

    # run test
    unlabeled = True
    results, accuracy = test_biomed_roberta(test_loader, classifier, args, extra=(iids, trials, orders, unlabeled))

    if accuracy is not None:
        print("Test accuracy of the model: {:2.3}".format(accuracy))
    else:
        print("Test accuracy of the model: N/A")
        generate_results_file(results, args, prefixes=['test_majority_', 'test_at_least_one_'])


if __name__ == '__main__':
    __args = parse_option()
    if __args.model_name[0:len(__MODEL_SLUG__)] != __MODEL_SLUG__:
        raise ValueError('Model name must be biomed')

    main_worker(__args)
