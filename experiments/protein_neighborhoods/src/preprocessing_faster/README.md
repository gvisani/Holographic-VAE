
## On the "faster" preprocessing

This folder contains a faster version of preprocessing, and a few differences in functionality over the `prerpocessing` folder. The main differences are:
1. In `preprocessing_faster` the projection is performed using numpy only instead of a mix of numpy and torch. This avoids the collisions between pytorch and multiprocessing that we were sometimes experiencing in `preprocessing`, resulting in much faster processing.
2. In `preprocessing_faster` we do not yet support the use of biopython to parse PDBs. This will be added soon. In the meantime, if you cannot use pyrosetta, we invite you to use `get_structural_info_pipeline.py` from `preprocessing`, and then switch to `preprocessing_faster` for the rest of the preprocessing pipeline.


## Extracting neighborhoods and computing zernike projections (zernikegrams) from a list of PDBs

The pipeline is divided in three steps, each with its corresponding script:

1. `get_structural_info.py` takes a list of PDBs as input and uses pyrosetta to parse them, saving structural information (atom coordinates, elements, SASA, partial charge, etc.) into an hdf5 file.
2. `get_neighborhoods.py` takes the hdf5 file generated in step 1 as input and constructs neighborhoods around each atom in the PDBs. The neighborhoods are saved into a new hdf5 file.
3. `get_zernikegrams.py` takes the hdf5 file generated in step 2 as input and constructs zernikegrams for each neighborhood. The zernikegrams are saved into a new hdf5 file.

Each script also bears a function, by the same name, that allows users to run the steps individually within other scripts. Import them simply by running:

```python
    from protein_holography_pytorch.preprocessing_faster import get_structural_info, get_neighborhoods, get_zernikegrams
```

Also, `pipeline.py` bears a function named `get_zernikegrams_from_structures()` that takes as input a list of structures (either as PDB files or pyrosetta poses), as well kwargs for `get_neighborhoods()` and `get_zernikegrams()`, and returns zernikegrams, as outputted by `get_zernikegrams()`.

When processing large volumes of proteins, we recommend using the scripts to leverage multiprocessing for faster computation. Processing 1,200 PDBs takes 15 minutes on 25 cores and 96GB of RAM (less memory is probably fine, but we haven't tested it).

For some example bash/slurm scripts that run the entire pipeline, checkout `/protein_holography-pytorch/runtime/preprocessing_scripts`.

**Note:** the use of pyrosetta to parse PDBs is currently necessary for computing SASA and partial charge, which are used by H-CNN. It is **not** necessary for H-(V)AE, since our models are not trained with SASa and partial charge. We will soon add an option to skip pyrosetta and use only biopython to parse PDBs.


## TODO

1. Add biopython parsing (the `preprocessing` folder contains biopython parsing)
