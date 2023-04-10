# Team INF-UFRGS at SemEval-2023 Task 7: Supervised Contrastive Learning for Pair-level Sentence Classification and Evidence Retrieval

## Creating the environment
```
conda create -n evidencescl python=3.9
conda activate evidencescl
pip install -U pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers
pip install wget
```

## Fetch the data to train and test the model
```
fetch_data.py [-h] [--dataset_url DATASET_URL]
              [--target_dir TARGET_DIR]
```

## Preprocess the data
```
preprocess_*.py [-h] [--config CONFIG]
```


## Train the encoder
```
python main_supcon.py  --epoch EPOCH --batch_size BatchSize --dataset Dataset --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0 
```

## Train the classifier
```
python main_validate.py --dataset Dataset --ckpt pathToModel --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0
```

## Test the model
```
python main_test.py --dataset Dataset --gpu GPU --ckpt_bert pathToEncoder --ckpt_classifier pathToClassifier
```

# Reference
If the code is used in your research, hope you can cite our paper as follows:
```
@inproceedings{CorreaDias+2023,
  author={Abel Corrêa Dias and }, 
  title={Team INF-UFRGS at SemEval-2023 Task 7: Supervised Contrastive Learning for Pair-level Sentence Classification and Evidence Retrieval}, 
  year={2023}
}
  ```
