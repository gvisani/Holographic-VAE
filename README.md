# Holographic-VAE

## Installation

Create the `hvae` conda environment by running the following

```bash
conda env create -f env.yml
```

to install the necessary dependencies.
Then run

```bash
pip install .
```

to install the `holographic_vae` package.

If you're going to make edits to the `holographic_vae` package, run

```bash
pip install -e .
```

so you can test your changes.

## Overview

### Holographic-(V)AE
Code containing building blocks for building expressive SO(3)-equivariant networks using spherical harmonics projections. The code is heavily based on the `e3nn` package. We decided to re-implement certain operations to work with *Dictionary* inputs (index by $\ell$) instead of *flat Tensor* inputs, as we observed speed gains during early development, as well as more flexibility to build custom normalization layers. The architecture of H-(V)AE is found in `holographic_vae/models/h_vae.py`.

### Experiments

#### MNIST-on-the-Sphere

Code replicating (most of) our experiments on the MNIST-on-the-Sphere dataset is found in `experiments/mnist`. Crucially, this folder contains general functions and scripts for forward and inverse spherical fourier transforms, helpful to preprocess any spherical image dataset to be used by H-(V)AE.

#### Protein Neighborhoods
Code to use H-(V)AE on protein neighborhoods. We provide code to use our pre-trained models to extract embeddings from a series of PDBs, as well as code to train new models. The code is found in `experiments/protein_neighborhoods`.

#### Ligand Binding Affinity
Code to replicate our experiments on the ATOM3D Ligand Binding Affinity task, where our method achieves state-of-the-art performance, as shown in our paper.  
**Note:** this necessitates having a trained H-(V)AE model, either provided by us, or user-trained. Checkout the README in `experiments/protein_neighborhoods` to learn how to download a pre-trained model, or train a new one with your data.

---

@misc{visani_holographic-vae_2023,  
  title = {Holographic-({V}){AE}: an end-to-end {SO}(3)-{Equivariant} ({Variational}) {Autoencoder} in {Fourier} {Space}},  
  author = {Visani, Gian Marco and Pun, Michael N. and Angaji, Arman and Nourmohammad, Armita},  
	year = {2023},  
  publisher = {bioRxiv},  
  doi = {10.1101/2022.09.30.510350},  
  url = {https://www.biorxiv.org/content/10.1101/2022.09.30.510350v2}  
 }


