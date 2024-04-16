from transformers import PreTrainedTokenizer, AddedToken, PreTrainedTokenizerBase
import numpy as np
from typing import List, Optional
import torch
import os
import collections


class GeneTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_dir, num_bins, **kwargs):
        self.cell_type_tokens = None
        self.tissue_type_tokens = None
        self.age_type_tokens = None
        self.sex_type_tokens = None
        self.disease_type_tokens = None

        self.start_token = "[START]"
        self.end_token = "[END]"
        self.no_sex_token = "[no_sex]"
        self.no_tissue_token = "[no_tissue]"
        self.no_cell_type_token = "[no_cell_type]"
        self.no_age_token = "[no_age]"
        self.no_disease_token = "[no_disease]"

        # Load and set the vocabulary
        self.phenotypic_token_names = self.load_phenotypic_tokens(vocab_dir)
        self.vocab = self.load_vocab(vocab_dir)

        self.num_bins = num_bins
        self.bin_vocab = {str(i): i+len(self.vocab) for i in range(num_bins)}
        self.vocab = self.vocab | self.bin_vocab
        self.idx2token = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.token2idx = collections.OrderedDict([(tok, ids) for tok, ids in self.vocab.items()])

        super().__init__(
            pad_token="[PAD]",
            mask_token="[MASK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            unk_token="[UNK]",
            **kwargs)

        self.add_special_tokens({"additional_special_tokens": [self.start_token,
                                                               self.end_token,
                                                               self.no_sex_token,
                                                               self.no_tissue_token,
                                                               self.no_cell_type_token,
                                                               self.no_age_token,
                                                               self.no_disease_token,
                                                               ]})


    @property
    def phenotypic_tokens(self):
        return self.phenotypic_token_names

    @property
    def phenotypic_tokens_ids(self) -> List[int]:
        return [self.token2idx[token] for token in self.phenotypic_token_names]

    @property
    def special_tokens_ids(self) -> List[int]:
        return [self.cls_token_id,
                self.sep_token_id,
                self.pad_token_id,
                self.mask_token_id,
                self.unk_token_id,
                self.start_token_id,
                self.end_token_id,
                self.no_sex_token_id,
                self.no_tissue_token_id,
                self.no_cell_type_token_id,
                self.no_age_token_id,
                self.no_disease_token_id
                ]

    def load_phenotypic_tokens(self, vocab_dir: str) -> List[str]:
        # load tissue type tokens
        self.tissue_type_tokens = []
        with open(os.path.join(vocab_dir, "tissue_types.txt"), 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.tissue_type_tokens.append('[' + token.replace(' ', '_').strip() + ']')

        # load cell type tokens
        self.cell_type_tokens = []
        with open(os.path.join(vocab_dir, "cell_types.txt"), 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.cell_type_tokens.append('[' + token.replace(' ', '_').strip() + ']')

        # load age type tokens
        self.age_type_tokens = []
        with open(os.path.join(vocab_dir, "age_types.txt"), 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.age_type_tokens.append('[' + token.replace(' ', '_').strip() + ']')

        # load sex type tokens
        self.sex_type_tokens = []
        with open(os.path.join(vocab_dir, "sex_types.txt"), 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.sex_type_tokens.append('[' + token.replace(' ', '_').strip() + ']')

        # load disease type tokens
        self.disease_type_tokens = []
        with open(os.path.join(vocab_dir, "disease_types.txt"), 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.disease_type_tokens.append('[' + token.replace(' ', '_').strip() + ']')

        phenotypic_vocab = []
        all_tokens = self.tissue_type_tokens + self.cell_type_tokens + self.age_type_tokens + self.sex_type_tokens + self.disease_type_tokens
        for token in all_tokens:
            phenotypic_vocab.append(token)

        return phenotypic_vocab

    def load_vocab(self, vocab_dir):
        vocab = {}
        if not os.path.exists(vocab_dir):
            raise ValueError(f"Phenotypic vocab directory {vocab_dir} does not exist.")
        with open(os.path.join(vocab_dir, "vocab.txt"), 'r', encoding='utf-8') as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index

        return vocab

    @property
    def not_expressed_id(self):
        return self.vocab['0']

    def phenotypic_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.phenotypic_tokens}

    def tissue_type_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.tissue_type_tokens}

    def age_type_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.age_type_tokens}

    def cell_type_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.cell_type_tokens}

    def sex_type_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.sex_type_tokens}

    def disease_type_label2id(self):
        return {key: id_ for key, id_ in self.vocab.items() if key in self.disease_type_tokens}

    def get_vocab(self):
        vocab = dict(self.added_tokens_encoder, **self.token2idx)
        return vocab

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def sep_token_id(self):
        return self.vocab[self.sep_token]

    @property
    def cls_token_id(self):
        return self.vocab[self.cls_token]

    @property
    def start_token_id(self):
        return self.vocab[self.start_token]

    @property
    def end_token_id(self):
        return self.vocab[self.end_token]

    @property
    def no_sex_token_id(self):
        return self.vocab[self.no_sex_token]

    @property
    def no_tissue_token_id(self):
        return self.vocab[self.no_tissue_token]

    @property
    def no_cell_type_token_id(self):
        return self.vocab[self.no_cell_type_token]

    @property
    def no_age_token_id(self):
        return self.vocab[self.no_age_token]

    @property
    def no_disease_token_id(self):
        return self.vocab[self.no_disease_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_phenotypic_tokens_mask(self, token_ids):
        special_tokens_mask = [1 if token_id in self.phenotypic_tokens_ids else 0 for token_id in token_ids]
        return special_tokens_mask

    def _encode_one(self, expressions: List[int],
                    sex: Optional[str] = None,
                    tissue_type: Optional[str] = None,
                    cell_type: Optional[str] = None,
                    age_type: Optional[str] = None,
                    disease_type: Optional[str] = None,
                    add_special_tokens: bool = True
                    ) -> List[int]:
        # convert to token names
        if sex:
            sex = f"[{sex.replace(' ', '_')}]"
        if tissue_type:
            tissue_type = f"[{tissue_type.replace(' ', '_')}]"
        if cell_type:
            cell_type = f"[{cell_type.replace(' ', '_')}]"
        if age_type:
            age_type = f"[{age_type.replace(' ', '_')}]"
        if disease_type:
            disease_type = f"[{disease_type.replace(' ', '_')}]"

        encoded_expressions = [self.vocab[str(expression)] for expression in expressions]

        if add_special_tokens:
            sex_token_id = self.vocab[sex] if sex else self.vocab['[no_sex]']
            tissue_type_token_id = self.vocab[tissue_type] if tissue_type else self.vocab['[no_tissue]']
            cell_type_token_id = self.vocab[cell_type] if cell_type else self.vocab['[no_cell_type]']
            age_type_token_id = self.vocab[age_type] if age_type else self.vocab['[no_age]']
            disease_type_token_id = self.vocab[disease_type] if disease_type else self.vocab['[no_disease]']
            batch_token_ids = [self.cls_token_id,
                               sex_token_id,
                               tissue_type_token_id,
                               cell_type_token_id,
                               age_type_token_id,
                               disease_type_token_id,
                               self.start_token_id,
                               *encoded_expressions,
                               self.end_token_id]
        else:
            batch_token_ids = [*encoded_expressions]

        return batch_token_ids

    def encode(self, expressions: List[int],
               sexes: Optional[List[str]] = None,
               tissue_types: Optional[List[str]] = None,
               cell_types: Optional[List[str]] = None,
               age_types: Optional[List[str]] = None,
               disease_types: Optional[List[str]] = None,
               return_tensors: str = "pt",
               add_special_tokens: bool = True,
               **kwargs):

        batch_token_ids = []
        if sexes is None:
            sexes = [None] * len(expressions)

        if tissue_types is None:
            tissue_types = [None] * len(expressions)

        if cell_types is None:
            cell_types = [None] * len(expressions)

        if age_types is None:
            age_types = [None] * len(expressions)

        if disease_types is None:
            disease_types = [None] * len(expressions)

        for batch, sex, tissue_type, cell_type, age_type, disease_type in zip(expressions, sexes, tissue_types,
                                                                              cell_types, age_types, disease_types):
            batch_token_ids.append(self._encode_one(batch,
                                                    sex,
                                                    tissue_type,
                                                    cell_type,
                                                    age_type,
                                                    disease_type,
                                                    add_special_tokens))

        if return_tensors == "np":
            batch_token_ids = np.array(batch_token_ids)
        elif return_tensors == "pt":
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)

        else:
            raise ValueError("The input must be a string or a list of strings.")

        return batch_token_ids


if __name__ == "__main__":
    num_bins = 5
    tokenizer = GeneTokenizer('data/phenotypic_vocab', num_bins=num_bins)
    tokenizer.tissue_type_label2id()
    print(tokenizer.all_special_tokens)
    print(len(tokenizer))
    print(tokenizer.vocab)
    print(tokenizer.token2idx)
    print(tokenizer.idx2token)
    print(tokenizer.pad_token_id)
    print(tokenizer.mask_token_id)
    print(tokenizer.unk_token_id)
    print(tokenizer.sep_token_id)
    print(tokenizer.cls_token_id)
    print(tokenizer.vocab_size)
    print(tokenizer.encode([1, 2, 3, 4, 5], add_special_tokens=True))
    print(tokenizer.encode([1, 2, 3, 4, 5], add_special_tokens=False))
    print(tokenizer.get_vocab())
