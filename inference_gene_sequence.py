import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from transformers import (
    BertConfig,
    Trainer,
    TrainingArguments,
    set_seed, EvalPrediction,
)
import os
import numpy as np

import wandb

from data_utils.dadc_dataset import get_dataset
from gene_bert import GeneBertForMaskedLM
from gene_tokenizer import GeneTokenizer
from utils.utils import load_pretraining_config, parse_args
from utils.data_collators import DataCollatorForLanguageModeling


def calculate_confusion_matrix(predictions, labels, num_labels):
    conf_matrix = np.zeros((num_labels, num_labels))
    for true, pred in zip(labels, predictions):
        # fixing the offset, first the bins and then the labels
        if true in tokenizer.bin_vocab.values():
            true = true - tokenizer.vocab['0']
        if pred in tokenizer.bin_vocab.values():
            pred = pred - tokenizer.vocab['0']

        if true in label2id.values():
            true = true - min(list(label2id.values())) + config.num_bins
        if pred in label2id.values():
            pred = pred - min(list(label2id.values())) + config.num_bins

        if true not in range(num_labels) or pred not in range(num_labels):
            print(f"true: {true}, pred: {pred}")
            continue

        conf_matrix[true, pred] += 1

    return conf_matrix


def compute_metrics(p: EvalPrediction):
    """
    Computes MLM accuracy from EvalPrediction object.

    Args:
    - p (EvalPrediction): An object containing the predictions and labels.

    Returns:
    - dict: A dictionary containing the accuracy under the key 'accuracy'.
    """
    # Extract predictions and labels from the EvalPrediction object
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    # Compute predictions: Take the argmax (highest probability) as the predicted token ID
    predictions = np.argmax(logits, axis=-1)

    # Flatten everything to compare easily, ignoring -100 used for non-masked tokens
    mask = (labels != -100) & (labels != tokenizer.not_expressed_id)
    predictions_flat = predictions[mask].flatten()
    labels_flat = labels[mask].flatten()

    # Compute accuracy
    accuracy_all = accuracy_score(labels_flat, predictions_flat)

    # we ignore the 0 as it is the padding token (not expressed)
    labels_names = [f'{i}' for i in range(1, num_bins)]
    labels_names += [x.replace('[', '').replace(']', '').replace('_', ' ') for x in list(label2id.keys())]

    conf_matrix = calculate_confusion_matrix(predictions_flat, labels_flat, len(labels_names))
    # labels = [f'{i}' for i in range(conf_matrix.shape[0])]
    df_cm = pd.DataFrame(conf_matrix, index=labels_names, columns=labels_names)
    # # Save DataFrame to CSV
    df_cm.to_csv(f'{working_dir}/confusion_matrix_{trainer.state.global_step}.csv')

    # Calculate precision and recall
    # Note: Set average='macro' for macro-average (average over classes)
    # Use zero_division=0 to handle cases where there is no true or predicted samples for a class
    precision_all = precision_score(labels_flat, predictions_flat, average='macro', zero_division=0)
    recall_all = recall_score(labels_flat, predictions_flat, average='macro', zero_division=0)

    # ignore the first 7 tokens (special tokens)
    predictions = predictions[:, 7:]
    labels = labels[:, 7:]

    predictions_flat = predictions[labels != -100].flatten()
    labels_flat = labels[labels != -100].flatten()

    accuracy = accuracy_score(labels_flat, predictions_flat)
    precision = precision_score(labels_flat, predictions_flat, average='macro', zero_division=0)
    recall = recall_score(labels_flat, predictions_flat, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "accuracy_all": accuracy_all,
        "precision_all": precision_all,
        "recall_all": recall_all
    }


if __name__ == "__main__":
    # Set up the Trainer
    set_seed(42)

    config = parse_args()

    # the input sequence has the following format:
    # [CLS] [SEX_TOKEN] [TISSUE_TYPE_TOKEN] [CELL_TYPE_TOKEN] [AGE_TYPE_TOKEN] [DISEASE_TYPE_TOKEN] [START] gene1 gene2 gene3 ... geneN [END]

    assert config.max_length == config.n_highly_variable_genes + 8, ("max_length must be equal to "
                                                                     "n_highly_variable_genes + 8, because we add 8 "
                                                                     "special tokens to the input_ids")


    # the naming convention for the pretraining is as follows:
    # {task}_{approach}_{model_name}_{max_length}
    # where task is either "pretrained" or "finetuned"(for classification or clustering)
    # where approach is either "binary" or "binning"
    # model_name is either "bert" or "roberta"
    # max_length is the maximum length of the input sequence
    # os.environ["WANDB_MODE"] = "disabled"  # turn off wandb
    os.environ["WANDB_PROJECT"] = "test_inference"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    # sample vocab
    num_bins = config.num_bins
    tokenizer = GeneTokenizer(config.phenotypic_tokens_file, num_bins)

    all_label2id = tokenizer.phenotypic_label2id()

    label2id = all_label2id

    train_dataset = get_dataset(data_path=config.train_data_path,
                                batch_size=config.train_batch_size,
                                tokenizer=tokenizer,
                                binary_expression=False,
                                num_bins=num_bins,
                                max_length=config.max_length,
                                threshold=config.threshold,
                                n_highly_variable_genes=config.n_highly_variable_genes,
                                sequence_types=config.sequence_types,
                                shard_size=config.shard_size,
                                shuffle=True)

    eval_dataset = get_dataset(data_path=config.eval_data_path,
                               batch_size=config.eval_batch_size,
                               tokenizer=tokenizer,
                               binary_expression=False,
                               num_bins=num_bins,
                               max_length=config.max_length,
                               threshold=config.threshold,
                               n_highly_variable_genes=config.n_highly_variable_genes,
                               sequence_types=config.sequence_types,
                               shard_size=config.shard_size,
                               shuffle=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=config.mlm_probability,
                                                    phenotypic_mlm=True,
                                                    phenotypic_mlm_probability=config.phenotypic_mlm_probability)

    # Set device
    device = 'cuda:0'

    # Set up the model
    # if config.model_name == "bert":
    bert_config = BertConfig(vocab_size=len(tokenizer.vocab),
                             max_position_embeddings=config.max_length,
                             #hidden_size=config.hidden_size,
                             # num_hidden_layers=config.num_hidden_layers,
                             # num_attention_heads=config.num_attention_heads,
                             )

    model = GeneBertForMaskedLM.from_pretrained(config.model_name).to(device)

    working_dir = f"{config.output_dir}_{config.n_highly_variable_genes}"

    training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        logging_dir=working_dir,
        dataloader_num_workers=20,
        logging_steps=10,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=int(config.save_steps * len(train_dataset)),
        save_total_limit=5,
        eval_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=int(0.25 * len(train_dataset)),
        eval_accumulation_steps=1,
        load_best_model_at_end=False,
        fp16=True,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    trainer.evaluate()

    wandb.finish()
