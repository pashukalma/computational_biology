### Machine learning and bioinformatics 
 
Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) to generate synthetic images (specifically for cancer datasets) and design new molecular structures for drug discovery. 

Biological Classification & Sequencing: Applying supervised learning models (Random Forest, Naive Bayes, etc.) to classify medical data such as breast cancer markers, toxicity in chemical compounds, and single-cell protein annotations. It also includes genomic sequencing analysis specifically for COVID-19.

#### Medical & Biological Data Science Workspace
This repository contains a collection of modules focused on machine learning applications in oncology, drug discovery, and genomics.

- Molecular Generation & GANs: - AI_GAN_VAE.ipyb explores the architectures of Generative Adversarial Networks (GAN) and Variational Autoencoders (VAE) for data synthesis. - AI_Molecule_Generation_VAE.ipynb, the implementation of VAEs for automatic chemical design and drug discovery (MolGAN approach). - AI_Synthetic_Images_Cancer_Dataset.ipynb: Generates synthetic histopathologic images using GANs to augment cancer detection datasets.

- Bioinformatics & Sequencing: - AI_Sequencing_coid19.ipynb processes COVID-19 genomic data, analyzing dinucleotides and motifs using Biopython. - Classification_SingleCellProteine.ipynb:, supervised learning for classifying single-cell protein data into functional categories (e.g., HSPC, Prog).

- Medical Classification & Analytics: - Classification_BreastCancer.ipynb, predictive modeling on breast cancer datasets using Random Forest and Naive Bayes classifiers. - Classification_Toxicity_Dataset.ipynb, analysis of chemical toxicity based on molecular descriptors (LogP, Molecular Weight, etc.). - Dimensionality_Reduction.ipynb:, techniques for reducing feature complexity in high-dimensional biological datasets. - Predictive_Modeling_PyTorch.ipynb: Deep learning implementation for predictive modeling on lung cancer datasets.

- Image Processing: - image_classifier.py, a standalone Python script utilizing Transfer Learning and Convolutional Neural Networks (CNN) via PyTorch Lightning to detect cancer in histopathologic images.