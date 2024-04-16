# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import re
import shutil
import tempfile
import urllib.request

import scanpy as sc
from anndata import AnnData, read_h5ad
from google.cloud.storage import Client

url_schemes = ("http:", "https:", "ftp:")


import requests
import anndata
import io


def read_h5ad_cloudflare(file_url: str) -> AnnData:
    # CloudFlare url where the data is stored:
    # file_url = 'https://pub-8978012207224952a747e641910bcb1c.r2.dev/ranked/Tabula_Sapiens_ranked_0.h5ad'

    # Fetch the file
    response = requests.get(file_url)
    if response.status_code == 200:
        # Create a BytesIO buffer from the binary content of the response
        with io.BytesIO(response.content) as f:
            # Load the AnnData object directly from the buffer
            return anndata.read_h5ad(f)
    else:
        print("Failed to fetch the AnnData file.")


def read_h5ad_gcs(filename: str, storage_client: Client | None = None) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
    """
    assert filename.startswith("gs:")
    # parse bucket and blob names from the filename
    filename = re.sub(r"^gs://?", "", filename)
    bucket_name, blob_name = filename.split("/", 1)

    if storage_client is None:
        storage_client = Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("rb") as f:
        return read_h5ad(f)


def read_h5ad_url(filename: str) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the URL.

    Example::

        >>> adata = read_h5ad_url(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
        ... )

    Args:
        filename: URL of the data file.
    """
    assert any(filename.startswith(scheme) for scheme in url_schemes)
    with urllib.request.urlopen(filename) as response:
        with tempfile.TemporaryFile() as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            return read_h5ad(tmp_file)


def read_h5ad_local(filename: str) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the local disk.

    Args:
        filename: Path to the local data file.
    """
    assert filename.startswith("file:")
    filename = re.sub(r"^file://?", "", filename)
    return read_h5ad(filename)


def read_h5ad_file(filename: str, n_highly_variable_genes=None, **kwargs) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Args:
        filename: Path to the data file.
        n_highly_variable_genes: top highly variable genes to select
    """
    if filename.startswith("gs:"):
        adata = read_h5ad_gcs(filename, **kwargs)

    elif filename.startswith("file:"):
        adata = read_h5ad_local(filename)

    elif re.match("(https):\/\/[a-zA-Z0-9\-\.]*\.(r2\.dev)", filename) is not None:
        adata = read_h5ad_cloudflare(filename)

    elif any(filename.startswith(scheme) for scheme in url_schemes):
        adata = read_h5ad_url(filename)
    else:
        adata = read_h5ad(filename)

    if n_highly_variable_genes:
        ## this one is using the default highly_variable genes from the adata

        if "highly_variable_rank" in adata.var.columns:
            # seurat_v3 which has the rank of the genes
            # we also sort the genes based on the rank from the lowest to the highest rank
            # or from the highest variable to the lowest variable
            high_ranked_indices = adata.var['highly_variable_rank'] < n_highly_variable_genes
            subset_genes = adata.var[high_ranked_indices].sort_values(by='highly_variable_rank').index
        else:
            # seurat which has the boolean highly_variable
            subset_genes = adata.var[adata.var['highly_variable']].index

        adata = adata[:, subset_genes]

    return adata
