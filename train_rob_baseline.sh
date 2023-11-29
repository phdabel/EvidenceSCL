#!/bin/bash


# train_rob_baseline.sh <batch_size> <gradient_accumulation_steps> <epochs> <workers>
BATCH_SIZE=${1:-8}
GRADIENT_ACCUMULATION_STEPS=${2:-1}
EPOCHS=${3:-5}
WORKERS=${4:-1}

echo "Running baseline model with batch size $BATCH_SIZE, gradient accumulation steps $GRADIENT_ACCUMULATION_STEPS, epochs $EPOCHS and workers $WORKERS"

for dataset in "rob"
do
  for learning_rate in 1e-5 3e-5 5e-5
  do
    for num_classes in 3
    do
      for max_seq_length in 128
      do
        echo "Dataset: $dataset, learning rate $learning_rate, $num_classes classes and $max_seq_length max_seq_length"
        python train_rob_baseline.py --dataset "$dataset" --max_seq_length "$max_seq_length" --num_classes "$num_classes" --model_name biomed --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --epochs "$EPOCHS" --evaluation_metric f1 --workers "$WORKERS" --learning_rate "$learning_rate"
      done
    done
  done
done
