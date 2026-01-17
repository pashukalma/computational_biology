GPU-Accelerated Clustering & Dimensionality Reduction (cuML)
This repository contains GPU workflows for data clustering and visualization. It utilizes RAPIDS cuML to provide significant speedups for common machine learning algorithms like UMAP and K-Means, compared to CPU-based implementations.

####**🚀 Overview**
High-dimensional data can be computationally expensive to process and difficult to visualize. This notebook explores:

UMAP (Uniform Manifold Approximation and Projection): Embedding high-dimensional data into 2D and 3D spaces for visual analysis.

K-Means Clustering: Partitioning data into distinct groups based on feature similarity.

GPU Acceleration: Leveraging NVIDIA GPUs and the cuML library to perform distance calculations and centroid updates in parallel.