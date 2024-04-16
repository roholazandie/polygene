# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import IterableDataset
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from .distributed_anndata import DistributedAnnDataCollection
from .util import get_rank_and_num_replicas, get_worker_info
import time
import logging

class IterableDistributedAnnDataCollectionDataset(IterableDataset):
    r"""
    Iterable DistributedAnnDataCollection Dataset.

    When :attr:`shuffle` is set to ``True`` then the iterator yields datapoints that are
    uniformly sampled from the entire dataset. Typical use cases include training variational
    models using the stochastic gradient descent algorithm.

    In order to maximize buffer usage, we only shuffle shards and datapoints within individual
    shards (and not across shards). Therefore, to achieve unbiased pseudo-random uniform sampling,
    it is imperative that the shards themselves contain datapoints that are uniformly sampled
    from the entire dataset. If correlations exist between datapoints in a given shard (e.g. all
    cells coming from the same tissue or experiment), then this assumption is violated. It is
    the user's responsibility to prepare appropriately shuffled data shards.

    Args:
        dadc:
            DistributedAnnDataCollection from which to load the data.
        batch_size:
            How many samples per batch to load.
        shuffle:
            If ``True``, the data is reshuffled at every epoch.
        seed:
            Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        drop_last:
            If ``True``, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas. If ``False``,
            the sampler will add extra indices to make the data evenly divisible across
            the replicas.
        test_mode:
            If ``True``, then tracking of cache and worker informations will be enabled.
    """

    def __init__(
        self,
        dadc: DistributedAnnDataCollection,
        batch_size: int = 1,
        shuffle: bool = False,
        max_position_embeddings: int = 512,
        tokenizer: PreTrainedTokenizer = None,
        binary_expression: bool = True,
        num_bins: int = 100,
        threshold: float = 0.5,
        label: str = None,
        label2id: dict[str, int] = None,
        sequence_types: list[str] = None,
        filter_phenotypes: dict[str, list[str]] = None,
        seed: int = 42,
        drop_last: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.dadc = dadc
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_position_embeddings = max_position_embeddings
        self.tokenizer = tokenizer
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if binary_expression:
            self.expression_threshold = threshold # values less than this indicates no expression
            self.bin_edges = np.array([0.0, self.expression_threshold])
            self.num_bins = 2
        else:
            # the binning is unequal, the first bin corresponds to no expression
            # is lower than the threshold, the rest of the bins are equal
            self.num_bins = num_bins - 1
            expression_min_value = 0.1
            expression_max_value = 9.0
            # self.bin_edges = np.linspace(expression_min_value, expression_max_value, num_bins)
            self.bin_edges = np.concatenate(([0.0], np.linspace(expression_min_value, expression_max_value, self.num_bins)))

        self.label = label if label else None
        self.label2id = label2id if label2id else None
        self.sequence_types = sequence_types
        self.num_classes = len(label2id) if label2id else None

        self.filter_phenotypes = filter_phenotypes

        self.test_mode = test_mode

    def __len__(self) -> int:
        """
        Returns the number of batches per replica.
        """
        _, num_replicas = get_rank_and_num_replicas()

        if self.drop_last and len(self.dadc) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = len(self.dadc) // num_replicas
        else:
            per_replica = math.ceil(len(self.dadc) / num_replicas)
        return math.ceil(per_replica / float(self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for the iterator. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch.
        """
        self.epoch = epoch

    def get_labels(self, idx, mask):
        tissue_labels = self.dadc[idx][mask].obs['tissue'].unique().tolist()
        cell_labels = self.dadc[idx][mask].obs['cell_type'].unique().tolist()
        age_labels = self.dadc[idx][mask].obs['development_stage'].unique().tolist()
        disease_labels = self.dadc[idx][mask].obs['disease'].unique().tolist()
        return tissue_labels, cell_labels, age_labels, disease_labels

    def __getitem__(self, idx: int | list[int] | slice) -> dict[str, np.ndarray]:
        r"""
        Returns a dictionary containing the data and metadata for the given index ``idx``.

        If the count data ``X`` is sparse then it is densified.
        """
        if self.filter_phenotypes:
            tissue_mask = pd.Series(True, index=self.dadc[idx].obs.df.index)
            if "tissue" in self.filter_phenotypes:
                tissue_mask = pd.Series(self.dadc[idx].obs['tissue']).isin(self.filter_phenotypes["tissue"])
                # get the labels of the tissues based on the mask
                # tissue_labels = self.dadc[idx][tissue_mask].obs['tissue'].unique().tolist()
                # cell_labels = self.dadc[idx][tissue_mask].obs['cell_type'].unique().tolist()
                # age_labels = self.dadc[idx][tissue_mask].obs['development_stage'].unique().tolist()
                # disease_labels = self.dadc[idx][tissue_mask].obs['disease'].unique().tolist()

            cell_type_mask = pd.Series(True, index=self.dadc[idx].obs.df.index)
            if "cell_type" in self.filter_phenotypes:
                cell_type_mask = pd.Series(self.dadc[idx].obs['cell_type']).isin(self.filter_phenotypes["cell_type"])

            age_mask = pd.Series(True, index=self.dadc[idx].obs.df.index)
            if "age" in self.filter_phenotypes:
                age_mask = pd.Series(self.dadc[idx].obs['development_stage']).isin(self.filter_phenotypes["age"])

            disease_mask = pd.Series(True, index=self.dadc[idx].obs.df.index)
            if "disease" in self.filter_phenotypes:
                disease_mask = pd.Series(self.dadc[idx].obs['disease']).isin(self.filter_phenotypes["disease"])

            mask = tissue_mask & cell_type_mask & age_mask & disease_mask
            if mask.sum() == 0:
                # if there is no cell left after filtering, return None
                return None

            X = self.dadc[idx][mask].X

            self.sequence_types = ["sex", "cell_type", "tissue", "age", "disease"]
        else:
            X = self.dadc[idx].X

        if self.sequence_types:
            obs_df = self.dadc[idx].obs.df
            sexes = obs_df["sex"].values.tolist() if "sex" in self.sequence_types else None
            cell_types = obs_df['cell_type'].values.tolist() if "cell_type" in self.sequence_types else None
            tissue_types = obs_df['tissue'].values.tolist() if "tissue" in self.sequence_types else None
            ages = obs_df['development_stage'].values.tolist() if "age" in self.sequence_types else None
            diseases = obs_df['disease'].values.tolist() if "disease" in self.sequence_types else None

        expression_data = torch.Tensor(X.toarray()) if issparse(X) else torch.Tensor(X)

        ndx = torch.arange(expression_data.shape[0], device=expression_data.device)
        input_x = expression_data[ndx, :]
        bin_indices = np.digitize(input_x, self.bin_edges) - 1


        input_ids = self.tokenizer.encode(bin_indices,
                                          sexes=sexes,
                                          tissue_types=tissue_types,
                                          cell_types=cell_types,
                                          age_types=ages,
                                          disease_types=diseases)

        # here we don't do padding because all of our sequences have the same length
        assert self.max_position_embeddings == input_ids.shape[1], ("max_position_embeddings must be equal to "
                                                                    "input_ids.shape[1] plus 8 for special tokens")

        # creating the attention mask that is 1 for all expressed genes and 0 for all non-expressed genes
        attention_mask = (~(input_ids == self.tokenizer.not_expressed_id)).long()

        if self.label:
            integer_labels = torch.tensor([self.label2id[label.replace(' ', '_')] for label in self.dadc[idx].obs[self.label].values.tolist()], dtype=torch.long)
        else:
            integer_labels = None

        if self.label:
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": integer_labels}
        else:
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    }

    def __iter__(self):
        r"""
        Iterate through the dataset by trying to minimize the amount of anndata files
        fetched by each worker.

        .. note::
            Returned iterator is determined by the ``torch.utils.data.get_worker_info()``
            and ``torch.distributed`` contexts. Iterated indices are evenly divided between replicas
            (see :attr:`drop_last`). If multiple workers per replica, then indices are further
            divided between workers (last worker might contain less indices than other workers, see
            examples below). Indices are shuffled and iterated in a manner that minimizes the overlap
            between the data chunks loaded by each worker.

        Example 1::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            n_obs=12
            num_replicas=1
            batch_size=2
            num_workers=3
            num_batches=6
            batches_per_worker=2
            per_worker=4

        +----------+-------+---------+
        |          |batch 0| batch 1 |
        +==========+=======+=========+
        | worker 0 | (0,1) | (2,3)   |
        +----------+-------+---------+
        | worker 1 | (4,5) | (6,7)   |
        +----------+-------+---------+
        | worker 2 | (8,9) | (10,11) |
        +----------+-------+---------+


        Example 2::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            num_replicas=1
            batch_size=2
            num_workers=2
            num_batches=6
            batches_per_worker=3
            per_worker=6

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,5) |
        +----------+-------+-------+-------+
        | worker 1 | (6,7) | (8,9) | (10,) |
        +----------+-------+-------+-------+


        Example 3::

            indices=[0, 1, 2, 3, 4, 5, 6, 7]
            n_obs=8
            num_replicas=1
            batch_size=3
            num_workers=2
            num_batches=3
            batches_per_worker=2
            per_worker=6

        +----------+---------+---------+
        |          | batch 0 | batch 1 |
        +==========+=========+=========+
        | worker 0 | (0,1,2) | (3,4,5) |
        +----------+---------+---------+
        | worker 1 | (6,7)   |         |
        +----------+---------+---------+


        Example 4::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            num_replicas=2
            drop_last=True

            truncated_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            total_size=10

            first_replica=[0, 1, 2, 3, 4]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

            second_replica=[5, 6, 7, 8, 9]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

        *Replica 1*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,)  |
        +----------+-------+-------+-------+

        *Replica 2*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (5,6) | (7,8) | (9,)  |
        +----------+-------+-------+-------+


        Example 5::

            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n_obs=11
            num_replicas=2
            drop_last=False

            padded_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
            total_size=12

            first_replica=[0, 1, 2, 3, 4, 5]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

            second_replica=[6, 7, 8, 9, 10, 0]
            batch_size=2
            num_workers=1
            num_batches=3
            batches_per_worker=3
            per_worker=6

        *Replica 1*

        +----------+-------+-------+-------+
        |          |batch 0|batch 1|batch 2|
        +==========+=======+=======+=======+
        | worker 0 | (0,1) | (2,3) | (4,5) |
        +----------+-------+-------+-------+

        *Replica 2*

        +----------+-------+-------+--------+
        |          |batch 0|batch 1|batch 2 |
        +==========+=======+=======+========+
        | worker 0 | (6,7) | (8,9) | (10,0) |
        +----------+-------+-------+--------+
        """
        if self.test_mode:
            # clear lru cache
            self.dadc.cache.clear()

        # replicas
        rank, num_replicas = get_rank_and_num_replicas()

        if self.drop_last and len(self.dadc) % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data.
            per_replica = len(self.dadc) // num_replicas
        else:
            per_replica = math.ceil(len(self.dadc) / num_replicas)
        total_size = per_replica * num_replicas
        batches_per_replica = math.ceil(per_replica / float(self.batch_size))

        # workers
        worker_id, num_workers = get_worker_info()

        batches_per_worker = math.ceil(batches_per_replica / float(num_workers))
        per_worker = batches_per_worker * self.batch_size

        # split workload
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, per_replica)

        # indices
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            iter_limits = list(zip([0] + self.dadc.limits, self.dadc.limits))
            # shuffle shards
            limit_indices = torch.randperm(len(iter_limits), generator=rng).tolist()
            indices = []
            for limit_idx in limit_indices:
                lower, upper = iter_limits[limit_idx]
                # shuffle cells within shards
                indices.extend((torch.randperm(upper - lower, generator=rng) + lower).tolist())
        else:
            indices = list(range(len(self.dadc)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        indices = indices[rank * per_replica : (rank + 1) * per_replica]
        assert len(indices) == per_replica

        yield from (self[indices[i: i + self.batch_size]] for i in range(iter_start, iter_end, self.batch_size))

        # Sets epoch for persistent workers
        self.set_epoch(self.epoch + 1)




def get_dataset(data_path,
                batch_size,
                tokenizer,
                binary_expression,
                num_bins,
                max_length,
                threshold,
                n_highly_variable_genes,
                sequence_types=None,
                filter_phenotypes=None,
                shuffle=False,
                shard_size=10000):
    dadc = DistributedAnnDataCollection(
        filenames=data_path,
        limits=None,
        shard_size=shard_size,
        last_shard_size=None,
        max_cache_size=4,
        cache_size_strictly_enforced=True,
        label=None,
        keys=None,
        index_unique=None,
        convert=None,
        indices_strict=True,
        obs_columns=None,
        n_highly_variable_genes=n_highly_variable_genes,
    )

    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc=dadc,
        batch_size=batch_size,
        shuffle=shuffle,
        max_position_embeddings=max_length,
        tokenizer=tokenizer,
        binary_expression=binary_expression,
        num_bins=num_bins,
        sequence_types=sequence_types,
        filter_phenotypes=filter_phenotypes,
        threshold=threshold,
        drop_last=True,
        test_mode=False,
    )

    return dataset