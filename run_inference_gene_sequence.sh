#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python inference_gene_sequence.py \
    --model_name "/home/rohola/codes/transformeromics/checkpoints/pretrained/binned/bert/all_features_checkpoints_2432/checkpoint-170200" \
    --phenotypic_tokens_file "data/phenotypic_vocab" \
    --n_epochs 100 \
    --train_batch_size 9 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --hidden_size 64 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --shard_size 10000 \
    --train_data_path "file:///home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..46}.h5ad" \
    --eval_data_path "file:///home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad" \
    --output_dir "checkpoints/pretrained/binned/bert/all_except_age_features_checkpoints" \
    --max_length 2440 \
    --n_highly_variable_genes 2432 \
    --save_steps 0.01 \
    --expression_max_value 10.0 \
    --expression_min_value 0.0 \
    --threshold 0.1 \
    --mlm_probability 0.15 \
    --phenotypic_mlm_probability 0.5 \
    --num_bins 10 \
    --sequence_types sex tissue cell_type disease \
    --device "cuda"
