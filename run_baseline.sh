#!/bin/bash

# mednli
# max_seq_length = 128|512
# epochs 5
# num_classes = 2-3
# ----------------------------------
# learning_rate = 1e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
# learning_rate = 3e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
# learning_rate = 5e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset mednli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset mednli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5

# multinli
# max_seq_length = 128|512
# num_classes = 2-3
# ----------------------------------
# learning_rate = 1e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
# learning_rate = 3e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
# learning_rate = 5e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset multinli --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset multinli --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5

# nli4ct
# max_seq_length = 128|512
# num_classes = 2-3
# ----------------------------------
# learning_rate = 1e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 1e-5
# learning_rate = 3e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 3e-5
# learning_rate = 5e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 3 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset nli4ct --max_seq_length 128 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5
python train_baseline.py --dataset nli4ct --max_seq_length 512 --num_classes 2 --model_name biomed --batch_size 32 --gradient_accumulation_steps 16 --epochs 5 --workers 2 --learning_rate 5e-5