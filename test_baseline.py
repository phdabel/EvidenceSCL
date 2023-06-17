import sys
import os
import pickle
import torch
import warnings
from util import parse_option, get_dataloaders, generate_results_file, compute_real_accuracy

import torch.backends.cudnn as cudnn
from transformers import RobertaTokenizer, RobertaModel
from models.linear_classifier import LinearClassifier
from pipeline.test_pipeline import run_classifier_test as test_biomed_roberta

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
    evaluate_dataset = args.evaluate_dataset if args.evaluate_dataset is not None else args.dataset
    stage = args.dataset_stage

    dataloader_struct = get_dataloaders(evaluate_dataset, args.data_folder, tokenizer, args.batch_size, args.workers,
                                        args.max_seq_length, args.num_classes)

    _loader = dataloader_struct['loader'][stage]
    iids = dataloader_struct['iids'][stage]
    trials = dataloader_struct['trials'][stage]
    orders = dataloader_struct['orders'][stage]
    genres = dataloader_struct['genres'][stage]
    unlabeled = True if evaluate_dataset == 'nli4ct' and stage == 'test' else False

    results, accuracy = test_biomed_roberta(_loader, classifier, args, extra=(iids, trials, orders, genres, unlabeled))

    _, _, _agg_results = compute_real_accuracy(results, unlabeled=unlabeled)

    # save results in pickle file
    with open(args.save_folder + '/test_results_' + args.model_name + '.pkl', 'wb') as f:
        pickle.dump(results, f)

    if accuracy is not None:
        print("Test accuracy of the model: {:2.3}".format(accuracy))
    else:
        print("Test accuracy of the model: N/A")

    generate_results_file(_agg_results, args, prefixes=['test_majority_', 'test_at_least_one_'])


if __name__ == '__main__':
    __args = parse_option()
    if __args.model_name[0:len(__MODEL_SLUG__)] != __MODEL_SLUG__:
        raise ValueError('Model name must be biomed')

    main_worker(__args)

    print("Done!")
    sys.exit(0)
