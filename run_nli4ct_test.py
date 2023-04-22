import os
import torch
import warnings
from util import parse_option, get_dataloader

from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier
from pipeline.test import run_test as test_biomed_roberta

warnings.filterwarnings("ignore")
__MODEL_SLUG__ = 'biomed'


def main_worker(args):
    model = RobertaModel.from_pretrained("allenai/biomed_roberta_base")
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")
    classifier = LinearClassifier(model, num_classes=args.num_classes)

    if args.encoder_ckpt is not None:
        # load encoder checkpoint
        encoder_ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
        model.load_state_dict(encoder_ckpt['encoder'])
        print("=> Loaded encoder checkpoint '{}' (epoch {})".format(args.encoder_ckpt, encoder_ckpt['epoch']))

    if args.ckpt is not None:
        classifier_ckpt = torch.load(args.ckpt, map_location='cpu')
        classifier.load_state_dict(classifier_ckpt['classifier'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.ckpt, classifier_ckpt['epoch']))
    else:
        try:
            classifier_ckpt = torch.load(os.path.join(args.save_folder, 'classifier_best.pth'), map_location='cpu')
            classifier.load_state_dict(classifier_ckpt['classifier'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.ckpt, classifier_ckpt['epoch']))
        except FileNotFoundError:
            raise "No classifier checkpoint found. Please specify a checkpoint to load or ensure a classifier"

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    # load test data
    test_loader, iids, trials, orders = get_dataloader(args.data_folder, args.dataset, "nli4ct_unlabeled_test.pkl",
                                                       tokenizer, args.batch_size, args.workers,
                                                       args.max_seq_length, test=True)

    # run test
    unlabeled = True
    accuracy = test_biomed_roberta(test_loader, model, classifier, args, extra=(iids, trials, orders, unlabeled))
    if accuracy is not None:
        print("Test accuracy of the model: {:2.3}".format(accuracy))
    else:
        print("Test accuracy of the model: N/A")


if __name__ == '__main__':
    __args = parse_option()
    if __args.model_name[0:len(__MODEL_SLUG__)] != __MODEL_SLUG__:
        raise ValueError('Model name must be biomed')

    main_worker(__args)
