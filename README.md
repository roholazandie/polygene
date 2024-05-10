[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) &nbsp;
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.5560007.svg&#41;]&#40;&#41;)
# Polygene
This is the official codebase for **Multimodal Learning for Mapping the Genotype-Phenotype Dynamics**


## Overview
In this codebase, we employ a novel method to encode gene expression data, 
considering all highly variable genes in a dataset. We organize the genes for 
each cell in the dataset in a fixed order, ranging from the most variable to the 
least variable (although this is simply a convenient convention and not strictly necessary).
Each gene is identified by its position in the `input_ids`.

We utilize a binning method to generate a continuous representation of the gene 
expression data, enabling finer-grained representations. In this context, the vocabulary
is defined by the number of bins for each gene, along with special tokens.

Additionally, we append special tokens to the input as follows:

1. [CLS] which is a special token that is used for classification tasks. This token is used to get the hidden state of the entire sequence.
2. [SEX_TYPE_TOKEN] which can get values of [NO_SEX] (not specified), [female], [male]
3. [TISSUE_TYPE_TOKEN] which can get values of [NO_TISSUE] or the name of the tissue.
4. [CELL_TYPE_TOKEN] which can get values of [NO_CELL_TYPE] or the name of the cell type.
5. [AGE_TYPE_TOKEN] which can get values of [NO_AGE] or the age of the individual.
6. [DISEASE_TYPE_TOKEN] which can get values of [NO_DISEASE] or the name of the disease.

Finally, the input to the model has the following format:

```
[CLS] [SEX_TYPE_TOKEN] [TISSUE_TYPE_TOKEN] [CELL_TYPE_TOKEN] [AGE_TYPE_TOKEN] [DISEASE_TYPE_TOKEN] [START] gene1 gene2 gene3 ... geneN [END]
```

These sequence type tokens helps with the tasks of denoising (masked language modeling) and classification
Also, they can be used as control tokens to create samples with different characteristics especially when
the dataset is imbalanced or low samples for some characteristics.
This approach has been fruitful in some models like T5 that uses control codes to generate samples with different characteristics.

### Loss function
In the case of binary encoding, we use the binary cross-entropy loss function. 
For binning, we employ the cross-entropy loss function with label smoothing. 
Label smoothing helps prevent the model from becoming overly confident in its predictions, which is particularly
important because the bins have a natural order (ordinal data). During training, the model should consider neighboring bins as well.




## Installation
Install gcc and g++:

```bash
sudo apt-get install gcc g++
```

Install the requirements:
```bash
pip install -r requirements.txt
```


## Pretrained Polygene models
Here is a list of pretrained models for Polygene of different sizes:

| Model           | Description                                               | Download Link                                                                                |
|:----------------|:----------------------------------------------------------|:---------------------------------------------------------------------------------------------|
| Polygene (2432) | Contains all the phenotypes and genotypes                 | [link](https://drive.google.com/file/d/1jmtOS3QfOpiGPE2OSsa_fQoiJfRntUcu/view?usp=drive_link) |
| Polygene (512)  | Smaller model containing all the phenotypes and genotypes | [link](https://drive.google.com/file/d/1Zj1OilSb4Dzoitx-Ycl7lrUgIo2VLbA0/view?usp=sharing) |
