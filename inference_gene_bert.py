from torch.utils.data import DataLoader
from transformers import BatchEncoding
from data_utils.dadc_dataset import get_dataset
from gene_bert import GeneBertForMaskedLM
from gene_tokenizer import GeneTokenizer
from utils.data_collators import DataCollatorForBatching, DataCollatorForControlledMasking, \
    DataCollatorForLanguageModeling
from utils.utils import load_inference_config
import torch
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def get_embeddings(config, n_samples=1000):
    tokenizer = GeneTokenizer(config.phenotypic_tokens_file, config.num_bins)
    # # Load the model
    model = GeneBertForMaskedLM.from_pretrained(config.pretrained_model_name_or_path)
    model.to(config.device)
    dataset = get_dataset(data_path=config.eval_data_path,
                          batch_size=config.eval_batch_size,
                          tokenizer=tokenizer,
                          binary_expression=config.binary_expression,
                          num_bins=config.num_bins,
                          max_length=config.max_length,
                          threshold=config.threshold,
                          n_highly_variable_genes=config.n_highly_variable_genes,
                          filter_phenotypes=config.filter_phenotypes,
                          shard_size=config.shard_size,
                          shuffle=False)

    data_collator = DataCollatorForControlledMasking(tokenizer=tokenizer,
                                                     masked_indices=[1])

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        # batch_size=14,
        num_workers=0,
        shuffle=False
    )

    all_embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(i)
            if len(batch) == 0:
                continue
            if i > n_samples:
                break
            model.eval()
            batch = batch.to(config.device)
            outputs = model(**batch, output_hidden_states=True)
            embeddings = list(torch.unbind(outputs.hidden_states[-1].detach().cpu(), dim=0))
            all_embeddings.extend(embeddings)

    return all_embeddings

import umap


if __name__ == "__main__":
    # for models that are trained on all phenotypes the samples should be taken
    # from a large dataset that contains all phenotypes that are not present in the filter_phenotypes keys
    # for models that are trained on one phenotype we have to set the value for
    # other phenotypes to no_phenotype such as no_sex, no_age, no_disease, no_tissue, no_cell_type

    # Load the phenotypes and gene list
    config = load_inference_config("configs/inference_binned_258.json")

    # Get the embeddings
    embeddings = get_embeddings(config, n_samples=10)

    embeddings_flat = np.array([tensor.numpy().flatten() for tensor in embeddings])
    embeddings_mean = np.array([tensor.mean(dim=1).numpy() for tensor in embeddings])

    # Step 2: Dimensionality Reduction with UMAP
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings_flat)

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], alpha=0.5)
    plt.title('2D UMAP Projection of the Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar()
    plt.savefig("paper_results/256_all_features/umap_sex_test.pdf", bbox_inches="tight")
    plt.show()

    ###### Plot the interaction matrices for cosine similarity and Mahalanobis distance ##########
    aggregated_embeddings = torch.mean(torch.stack(embeddings), dim=0)
    aggregated_embeddings_np = aggregated_embeddings.detach().numpy()
    num_genes = aggregated_embeddings.shape[0]

    # Calculate the interaction matrices for cosine similarity
    cosine_interaction_matrix = np.zeros((num_genes, num_genes))

    # Calculate the interaction matrices for Mahalanobis distance
    # For Mahalanobis, we need the inverse covariance matrix
    cov_matrix = np.cov(aggregated_embeddings_np.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_interaction_matrix = np.zeros((num_genes, num_genes))

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            # Cosine similarity
            cosine_similarity = np.dot(aggregated_embeddings_np[i], aggregated_embeddings_np[j]) / (
                        np.linalg.norm(aggregated_embeddings_np[i]) * np.linalg.norm(aggregated_embeddings_np[j]))
            cosine_interaction_matrix[i, j] = cosine_interaction_matrix[j, i] = cosine_similarity

            # Mahalanobis distance
            diff = aggregated_embeddings_np[i] - aggregated_embeddings_np[j]
            mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
            mahalanobis_interaction_matrix[i, j] = mahalanobis_interaction_matrix[j, i] = mahalanobis_distance

    # Plot the interaction matrices
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.viridis  # Define colormap

    # Plot for cosine similarity
    mask = np.triu(np.ones_like(cosine_interaction_matrix, dtype=bool))
    masked_cosine_matrix = np.ma.array(cosine_interaction_matrix, mask=mask)
    cax1 = axs[0].matshow(masked_cosine_matrix, cmap=cmap)
    fig.colorbar(cax1, ax=axs[0])
    axs[0].set_title("Cosine Similarity Matrix (Liver)")
    axs[0].set_xlabel("Gene Index")
    axs[0].set_ylabel("Gene Index")

    # Plot for Mahalanobis distance
    # For Mahalanobis, lower values mean more similarity, so we invert the colors
    mask = np.triu(np.ones_like(mahalanobis_interaction_matrix, dtype=bool))
    masked_mahalanobis_matrix = np.ma.array(mahalanobis_interaction_matrix, mask=mask)
    cax2 = axs[1].matshow(masked_mahalanobis_matrix, cmap=cmap.reversed())
    fig.colorbar(cax2, ax=axs[1])
    axs[1].set_title("Mahalanobis Distance Matrix (Liver)")
    axs[1].set_xlabel("Gene Index")
    axs[1].set_ylabel("Gene Index")

    plt.tight_layout()
    plt.savefig("paper_results/256_all_features/interaction_matrices_liver.pdf", bbox_inches="tight")
    plt.show()



    # Initialize a list to hold the mean embedding for each gene
    gene_embeddings = []

    # Loop over each gene position to calculate the mean embedding across all cells
    for gene_idx in range(20):  # Assuming 10 genes
        # Extract the embedding for this gene from each cell's embeddings
        gene_specific_embeddings = torch.stack([cell_embeddings[gene_idx] for cell_embeddings in embeddings])

        # Calculate the mean embedding for this gene across all cells
        mean_gene_embedding = torch.mean(gene_specific_embeddings, dim=0)

        # Add this mean embedding to the list of gene embeddings
        gene_embeddings.append(mean_gene_embedding)

    # Convert list of tensors to a single tensor for TSNE
    # gene_embeddings_tensor = torch.stack(gene_embeddings)
    # gene_embeddings_numpy = gene_embeddings_tensor.detach().numpy()

    # Calculate variance for each gene's embedding
    variances = [torch.var(embedding).item() for embedding in gene_embeddings]