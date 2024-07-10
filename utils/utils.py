import json
import argparse
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AnnTrainConfig:
    model_name: str
    phenotypic_tokens_file: str
    n_epochs: int
    train_batch_size: int
    eval_batch_size: int
    train_data_path: str
    eval_data_path: str
    output_dir: str
    save_steps: float
    device: str
    learning_rate: float = 1e-4
    hidden_size: int = 64
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    shard_size: int = 10000
    max_length: int = 512
    expression_max_value: float = 10.0
    expression_min_value: float = 0.0
    threshold: float = 0.5
    mlm_probability: float = 0.15
    phenotypic_mlm_probability: float = 0.50
    num_bins: int = 10
    n_highly_variable_genes: int = None
    sequence_types: list = None
    binary_expression: bool = False


@dataclass
class AnnClassificationConfig:
    pretrained_model_path: str
    special_tokens_file: str
    n_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    shard_size: int
    train_data_path: str
    eval_data_path: str
    summary_embeddings_file: str
    output_dir: str
    max_length: int
    n_highly_variable_genes: int
    save_steps: float
    expression_max_value: float
    expression_min_value: float
    threshold: float
    label: str
    num_bins: int
    binary_expression: bool
    device: str


@dataclass
class AnnInferenceConfig:
    pretrained_model_name_or_path: str
    phenotypic_tokens_file: str
    eval_batch_size: int
    train_data_path: str
    eval_data_path: str
    max_length: int
    shard_size: int
    n_highly_variable_genes: int
    expression_max_value: float
    expression_min_value: float
    threshold: float
    filter_phenotypes: Dict[str, List[str]]
    num_bins: int
    binary_expression: bool
    device: str


def load_inference_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return AnnInferenceConfig(**config_dict)

def load_pretraining_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return AnnTrainConfig(**config_dict)


def load_ann_classification_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return AnnClassificationConfig(**config_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Model training configuration")

    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--phenotypic_tokens_file", type=str, default="data/phenotypic_vocab",
                        help="File containing phenotypic tokens")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=9, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=80, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--shard_size", type=int, default=10000, help="Shard size")
    parser.add_argument("--train_data_path", type=str,
                        default="file:///home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..46}.h5ad",
                        help="Path to the training data")
    parser.add_argument("--eval_data_path", type=str,
                        default="file:///home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad",
                        help="Path to the evaluation data")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/pretrained/binned/bert/all_except_age_features_checkpoints",
                        help="Output directory")
    parser.add_argument("--max_length", type=int, default=2440, help="Maximum length")
    parser.add_argument("--n_highly_variable_genes", type=int, default=2432, help="Number of highly variable genes")
    parser.add_argument("--save_steps", type=float, default=0.01, help="Save steps")
    parser.add_argument("--expression_max_value", type=float, default=10.0, help="Expression max value")
    parser.add_argument("--expression_min_value", type=float, default=0.0, help="Expression min value")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM probability")
    parser.add_argument("--phenotypic_mlm_probability", type=float, default=0.5, help="Phenotypic MLM probability")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins")
    parser.add_argument("--sequence_types", type=str, nargs='+', default=["sex", "tissue", "cell_type", "disease"],
                        help="Sequence types")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()
