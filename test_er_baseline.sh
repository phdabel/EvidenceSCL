#!/bin/bash


# run_baseline.sh <batch_size> <gradient_accumulation_steps> <epochs> <workers>
BATCH_SIZE=${1:-8}
GRADIENT_ACCUMULATION_STEPS=${2:-1}
EPOCHS=${3:-5}
WORKERS=${4:-1}

echo "Running baseline model with batch size $BATCH_SIZE, gradient accumulation steps $GRADIENT_ACCUMULATION_STEPS, epochs $EPOCHS and workers $WORKERS"

for learning_rate in 1e-5 3e-5 5e-5
do
  for max_seq_length in 128 512
  do
    echo "Dataset: nli4ct, learning rate $learning_rate, $num_classes 2 and $max_seq_length max_seq_length"
    python test_er_baseline.py --dataset nli4ct --max_seq_length "$max_seq_length" --num_classes 2 --model_name biomed --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --epochs "$EPOCHS"  --workers "$WORKERS" --learning_rate "$learning_rate" --evidence_retrieval
  done
done