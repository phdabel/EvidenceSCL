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


Downloads the dataset and extracts its content into the target directory (Default value: `target_dir='datasets/raw'`).
Dataset argument accepts `NLI4CT`, `MultiNLI` or the URL to a custom dataset.

```
fetch_data.py [-h] [--dataset DATASET]
              [--target_dir TARGET_DIR]
```

## Preprocess the data

```
preprocess_*.py [-h] [--config CONFIG]
```


## Train the encoder
```
python train_encoder.py --model_name EvidenceSCL --dataset NLI4CT --max_seq_length 512 --batch_size 8 --gradient_accumulation_steps 64 
```

## Train the classifier
```
python train_classifier.py --model_name EvidenceSCL --dataset NLI4CT --max_seq_length 512 --batch_size 8 --gradient_accumulation_steps 64 --ckpt_encoder <encoder_checkpoint_path>
```

## Test the model

```
@todo
```

# Reference
If the code is used in your research, hope you can cite our paper as follows:
```
@inproceedings{CorreaDias+2023,
  author={Abel Corrêa Dias and Filipe Faria Dias and Higor Moreira and Viviane P. Moreira and João Luiz Dihl Comba}, 
  title={Team INF-UFRGS at SemEval-2023 Task 7: Supervised Contrastive Learning for Pair-level Sentence Classification and Evidence Retrieval},
  booktitle={Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)},
  publisher={Association for Computational Linguistics},
  year={2023}
}
  ```
