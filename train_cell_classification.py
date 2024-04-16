import time

from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import PreTrainedTokenizer, AddedToken, BertForSequenceClassification, \
    AutoModelForSequenceClassification, AutoConfig
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import os
import numpy as np
import wandb
from transformers import EvalPrediction
from data_utils import DistributedAnnDataCollection, IterableDistributedAnnDataCollectionDataset
from gene_tokenizer import GeneTokenizer
from utils.utils import load_config, load_pretraining_config, load_ann_classification_config
from utils.data_collators import DataCollatorForLanguageModeling, DataCollatorForBatching
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataset(data_path,
                batch_size,
                tokenizer,
                binary_expression,
                num_bins,
                threshold,
                n_highly_variable_genes,
                max_position_embeddings,
                label,
                label2id,
                shuffle=False,
                shard_size=10000):

    dadc = DistributedAnnDataCollection(
        filenames=data_path,
        limits=None,
        shard_size=shard_size,
        last_shard_size=None,
        max_cache_size=1,
        cache_size_strictly_enforced=True,
        label=None,
        keys=None,
        index_unique=None,
        convert=None,
        indices_strict=True,
        obs_columns=None,
        n_highly_variable_genes=n_highly_variable_genes
    )

    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc=dadc,
        batch_size=batch_size,
        shuffle=shuffle,
        max_position_embeddings=max_position_embeddings,
        tokenizer=tokenizer,
        binary_expression=binary_expression,
        num_bins=num_bins,
        threshold=threshold,
        label=label,
        label2id=label2id,
        drop_last=True,
        test_mode=False,
    )

    return dataset


def get_label_information(label):
    # tissue
    if label == "tissue":
        tissues = ['subcutaneous adipose tissue', 'small intestine', 'bladder organ', 'liver', 'blood',
                   'cardiac atrium', 'inguinal lymph node', 'spleen', 'muscle of pelvic diaphragm', 'adipose tissue',
                   'sclera', 'lung','kidney', 'uterus', 'skin of chest', 'endocrine pancreas', 'anterior part of tongue',
                   'exocrine pancreas', 'lymph node', 'coronary artery', 'endometrium', 'cornea', 'parotid gland',
                   'aorta', 'skin of abdomen', 'muscle of abdomen', 'thymus', 'eye', 'bone marrow', 'sublingual gland',
                   'cardiac ventricle', 'lacrimal gland', 'large intestine', 'posterior part of tongue', 'skin of body',
                   'prostate gland', 'muscle tissue', 'retinal neural layer', 'rectus abdominis muscle', 'tongue',
                   'mammary gland', 'conjunctiva', 'vasculature', 'trachea', 'myometrium']
        tissues = [tissue.replace(" ", "_") for tissue in tissues]
        id2label = {i: tissue for i, tissue in enumerate(tissues)}
        label2id = {tissue: i for i, tissue in enumerate(tissues)}

    elif label == "tissue_in_publication":
        tissue_in_publication = ['Liver', 'Bone_Marrow', 'Thymus', 'Fat', 'Spleen', 'Pancreas', 'Mammary', 'Eye',
                                 'Small_Intestine', 'Prostate',
                                 'Kidney', 'Salivary_Gland', 'Muscle', 'Tongue', 'Lung', 'Blood', 'Heart',
                                 'Vasculature', 'Trachea', 'Skin',
                                 'Large_Intestine', 'Lymph_Node', 'Uterus', 'Bladder']
        tissue_in_publication = [tissue.replace(" ", "_") for tissue in tissue_in_publication]
        id2label = {i: tissue for i, tissue in enumerate(tissue_in_publication)}
        label2id = {tissue: i for i, tissue in enumerate(tissue_in_publication)}

    elif label == "cell_type":
        cell_types = ['luminal epithelial cell of mammary gland', 'endothelial cell of vascular tree',
                      'pancreatic acinar cell', 'transit amplifying cell of small intestine',
                      'serous cell of epithelium of bronchus', 'lung microvascular endothelial cell',
                      'non-classical monocyte', 'myoepithelial cell', 'conjunctival epithelial cell',
                      'epithelial cell of uterus', 'fast muscle cell', 'myofibroblast cell',
                      'transit amplifying cell of colon', 'mesothelial cell', 'fibroblast of cardiac tissue',
                      'type II pneumocyte', 'slow muscle cell', 'endothelial cell of hepatic sinusoid',
                      'ciliated epithelial cell', 'Mueller cell', 'type I pneumocyte',
                      'skeletal muscle satellite stem cell', 'platelet', 'CD141-positive myeloid dendritic cell',
                      'tendon cell', 'intestinal crypt stem cell of large intestine', 'myeloid dendritic cell',
                      'gut endothelial cell', 'retinal ganglion cell', 'dendritic cell', 'retina horizontal cell',
                      'monocyte', 'effector CD8-positive, alpha-beta T cell', 'melanocyte', 'ciliated cell',
                      'intestinal enteroendocrine cell', 'myometrial cell', 'immature natural killer cell',
                      'pancreatic stellate cell', 'basal cell', 'DN1 thymic pro-T cell',
                      'pigmented ciliary epithelial cell', 'tongue muscle cell', 'mucus secreting cell', 'naive B cell',
                      'intestinal tuft cell', 'CD8-positive, alpha-beta T cell',
                      'retinal blood vessel endothelial cell', 'pancreatic ductal cell', 'cell of skeletal muscle',
                      'hematopoietic stem cell', 'keratocyte', 'Langerhans cell', 'mature conventional dendritic cell',
                      'duct epithelial cell', 'regulatory T cell', 'Schwann cell',
                      'double-positive, alpha-beta thymocyte', 'CD4-positive, alpha-beta memory T cell', 'club cell',
                      'stem cell', 'T follicular helper cell', 'native cell', 'type B pancreatic cell',
                      'CD8-positive, alpha-beta cytokine secreting effector T cell', 'ionocyte',
                      'naive thymus-derived CD8-positive, alpha-beta T cell', 'capillary endothelial cell',
                      'common myeloid progenitor', 'sperm', 'pancreatic PP cell', 'fibroblast', 'granulocyte',
                      'kidney epithelial cell', 'basophil', 'acinar cell of salivary gland', 'adventitial cell',
                      'salivary gland cell', 'plasmacytoid dendritic cell', 'intrahepatic cholangiocyte',
                      'naive regulatory T cell', 'memory B cell', 'epithelial cell',
                      'endothelial cell of lymphatic vessel', 'pulmonary ionocyte',
                      'CD8-positive, alpha-beta cytotoxic T cell', 'classical monocyte',
                      'CD4-positive, alpha-beta T cell', 'B cell', 'serous cell of epithelium of trachea',
                      'radial glial cell', 'paneth cell of epithelium of small intestine', 'pericyte',
                      'CD4-positive helper T cell', 'muscle cell', 'respiratory goblet cell',
                      'CD1c-positive myeloid dendritic cell', 'effector CD4-positive, alpha-beta T cell',
                      'vascular associated smooth muscle cell', 'smooth muscle cell', 'lung ciliated cell',
                      'bladder urothelial cell', 'vein endothelial cell', 'plasmablast', 'liver dendritic cell',
                      'small intestine goblet cell', 'epithelial cell of lacrimal sac', 'cardiac endothelial cell',
                      'innate lymphoid cell', 'enterocyte of epithelium of large intestine', 'mature NK T cell',
                      'retinal pigment epithelial cell', 'basal cell of prostate epithelium',
                      'erythroid progenitor cell', 'plasma cell', 'CD8-positive, alpha-beta memory T cell',
                      'neutrophil', 'hepatocyte', 'thymocyte', 'macrophage', 'paneth cell of colon',
                      'pancreatic A cell', 'large intestine goblet cell', 'fat cell', 'keratinocyte',
                      'pancreatic D cell', 'intestinal crypt stem cell', 'eye photoreceptor cell', 'stromal cell',
                      'duodenum glandular cell', 'cardiac muscle cell',
                      'naive thymus-derived CD4-positive, alpha-beta T cell', 'mesenchymal stem cell',
                      'erythroid lineage cell', 'erythrocyte', 'DN3 thymocyte', 'luminal cell of prostate epithelium',
                      'endothelial cell', 'corneal epithelial cell', 'retinal bipolar neuron', 'enterocyte',
                      'tracheal goblet cell', 'goblet cell', 'DN4 thymocyte', 'connective tissue cell',
                      'endothelial cell of artery', 'intestinal crypt stem cell of small intestine',
                      'bronchial smooth muscle cell', 'secretory cell', 'T cell', 'surface ectodermal cell',
                      'enterocyte of epithelium of small intestine', 'microglial cell', 'fibroblast of breast',
                      'medullary thymic epithelial cell', 'type I NK T cell', 'mast cell', 'leukocyte',
                      'intermediate monocyte', 'myeloid cell', 'blood vessel endothelial cell']

        cell_types = [cell_type.replace(" ", "_") for cell_type in cell_types]
        id2label = {i: cell_type for i, cell_type in enumerate(cell_types)}
        label2id = {cell_type: i for i, cell_type in enumerate(cell_types)}

    elif label == "developmental_stage":
        developmental_stage = ['40-year-old human stage', '33-year-old human stage',
                               '42-year-old human stage', '37-year-old human stage',
                               '38-year-old human stage', '46-year-old human stage',
                               '74-year-old human stage', '59-year-old human stage',
                               '69-year-old human stage', '67-year-old human stage',
                               '57-year-old human stage', '61-year-old human stage',
                               '22-year-old human stage', '56-year-old human stage']
        developmental_stage = [stage.replace(" ", "_") for stage in developmental_stage]
        id2label = {i: stage for i, stage in enumerate(developmental_stage)}
        label2id = {stage: i for i, stage in enumerate(developmental_stage)}

    return id2label, label2id


def single_label_metrics(predictions, labels, threshold=0.5):
    # Convert predictions to a PyTorch tensor and apply sigmoid
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(torch.Tensor(predictions))

    # Convert probabilities to class predictions
    y_pred = torch.argmax(probs, dim=1).numpy()

    # Compute metrics
    y_true = labels
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    # roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    accuracy = accuracy_score(y_true, y_pred)

    all_f1 = f1_score(y_true, y_pred, average=None)
    print(f"all_f1: {all_f1}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    labels = [label.replace("_", " ") for label in list(id2label.values())]
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    # Save DataFrame to CSV
    df_cm.to_csv(f'{working_dir}/confusion_matrix_{trainer.state.global_step}.csv')

    # Plotting the confusion matrix
    plt.figure(figsize=(17, 17))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 6})  # Adjust font size as needed
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    # Adjust the position of the plot in the figure to give more space for labels
    plt.subplots_adjust(bottom=0.15, left=0.15)

    # Optionally, rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)

    # Save plot to file
    plt.savefig(f'{working_dir}/confusion_matrix_{trainer.state.global_step}.png')

    # Return metrics as a dictionary
    metrics = {'f1': f1,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = single_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


if __name__ == "__main__":
    # Set up the Trainer
    set_seed(42)
    config = load_ann_classification_config("configs/classification_binary_bert_258.json")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'configs/credentials.json'

    os.environ["WANDB_PROJECT"] = "cell_binary_classification"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    assert config.max_length == config.n_highly_variable_genes + 2, ("max_length must be equal to "
                                                                     "n_highly_variable_genes + 2, because we add 2 "
                                                                     "special tokens (CLS and SEP) to the input_ids")

    assert (config.binary_expression and config.num_bins == 2 or
            not config.binary_expression and config.num_bins > 2), ("num_bins must be 2 if binary_expression is True,"
                                                                    "otherwise num_bins must be greater than 2")

    # sample vocab
    num_bins = 2 if config.binary_expression else config.num_bins
    tokenizer = GeneTokenizer(num_bins)

    id2label, label2id = get_label_information(config.label)

    train_dataset = get_dataset(data_path=config.train_data_path,
                                batch_size=config.train_batch_size,
                                tokenizer=tokenizer,
                                binary_expression=config.binary_expression,
                                label=config.label,
                                label2id=label2id,
                                num_bins=num_bins,
                                threshold=config.threshold,
                                n_highly_variable_genes=config.n_highly_variable_genes,
                                max_position_embeddings=config.max_length,
                                shard_size=config.shard_size,
                                shuffle=True)

    eval_dataset = get_dataset(data_path=config.eval_data_path,
                               batch_size=config.eval_batch_size,
                               tokenizer=tokenizer,
                               binary_expression=config.binary_expression,
                               label=config.label,
                               label2id=label2id,
                               num_bins=num_bins,
                               threshold=config.threshold,
                               n_highly_variable_genes=config.n_highly_variable_genes,
                               max_position_embeddings=config.max_length,
                               shard_size=config.shard_size,
                               shuffle=False)

    # todo this should be removed in future: we don't need DataCollator for removing extra shape
    data_collator = DataCollatorForBatching(tokenizer=tokenizer)

    # Set device
    device = 'cuda:0'  # torch.device(config.device if torch.cuda.is_available() else "cpu")

    # # Load the configuration for a model
    model_config = AutoConfig.from_pretrained(config.pretrained_model_path,
                                              num_labels=len(id2label))

    model_config.id2label = id2label
    model_config.label2id = label2id

    model_config.problem_type = "single_label_classification"

    model = AutoModelForSequenceClassification.from_config(config=model_config)

    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {model_total_params}")

    working_dir = f"{config.output_dir}_{config.max_length}_classification_{config.label}"

    training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=False,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=config.learning_rate,
        logging_dir=working_dir,
        dataloader_num_workers=10,
        logging_steps=3,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=int(config.save_steps * len(train_dataset)),
        save_total_limit=5,
        evaluation_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=int(0.1 * len(train_dataset)),
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    wandb.finish()
