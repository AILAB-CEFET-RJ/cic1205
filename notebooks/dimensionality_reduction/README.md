# Dimensionality Reduction Companion Notebooks

These notebooks accompany `lecture_notes/15_dimensionality_reduction.pdf`.

## Sequence

1. `01_curse_of_dimensionality.ipynb`: sparsity, distance concentration, and k-NN.
2. `02_pca_math_background.ipynb`: data matrix, covariance, orthogonality, eigenvectors, and SVD.
3. `03_pca_from_scratch.ipynb`: manual PCA implementation, projection, reconstruction, and explained variance.
4. `04_pca_sklearn_and_model_selection.ipynb`: scikit-learn PCA, scree plots, thresholds, solvers, and whitening.
5. `05_pca_in_ml_pipeline.ipynb`: PCA inside pipelines, cross-validation, leakage, and predictive relevance.
6. `06_tsne_visualization.ipynb`: t-SNE, perplexity, random state, and plot interpretation.
7. `07_umap_visualization.ipynb`: UMAP hyperparameters and optional transform support.
8. `08_pca_tsne_umap_comparison.ipynb`: side-by-side comparison of PCA, t-SNE, and UMAP.

## Coverage Map

- Curse of dimensionality and k-NN: notebook 01.
- Feature selection, feature extraction, and embedding: notebooks 01 and 08.
- PCA mathematics and algorithm: notebooks 02, 03, and 04.
- Explained variance and component selection: notebooks 03, 04, and 05.
- PCA inside ML pipelines without leakage: notebook 05.
- t-SNE as local-neighborhood visualization: notebook 06.
- UMAP as graph/manifold neighborhood embedding: notebook 07.
- Careful interpretation and method choice: notebooks 06, 07, and 08.

## Dependency Note

The UMAP notebook requires `umap-learn` for UMAP-specific cells. It is optional; the notebook remains executable and explains how to install it when missing.
