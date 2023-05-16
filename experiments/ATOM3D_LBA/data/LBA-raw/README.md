# LBA: Ligand Binding Affinity


## Overview

In this task, we predict the binding affinity of ligands to their corresponding
proteins based on the co-crystallized structure of the protein-ligand complex.
We predict experimentally measured binding affinity as pK, defined as -log(Ki)
or -log(Kd), depending on which measurement is available.

We derive crystal structures and ligand binding data from PDBBind (Wang et al., 2004),
a widely-used curated database of protein-ligand complexes with experimental affinities
derived from literature. We use the 2019 update of the so-called "refined set", a subset
of complexes selected based on the quality of the structures and the affinity data.
After filtering ligands which could not be read by RDKit due to invalid bonding data,
our final dataset consists of 4,463 complexes.


## Datasets

- raw: the complete dataset
- splits:
   - split-by-sequence-identity-30: No proteins with seq. id. of more than 30% are in the same set. 
   - split-by-sequence-identity-60: No proteins with seq. id. of more than 60% are in the same set.


## Usage

To read structures and labels, use the atom3d package:

    from atom3d.datasets import LMDBDataset
    dataset = LMDBDataset(PATH_TO_LMDB)
    print(len(dataset))  # Print length
    print(dataset[0])  # Print 1st entry
    labels = [item['scores']['neglog_aff'] for item in dataset] # Get all labels


## Format

Each entry in the dataset contains the following keys:

['atoms_protein'] (pandas.DataFrame) All atoms of the protein.
['atoms_pocket'] (pandas.DataFrame) Only the binding pocket (protein atoms within 6.0 Angstroms
of any atom in the ligand).
['atoms_ligand'] (pandas.DataFrame) Atoms of the ligand.
['bonds'] (pandas.DataFrame) Bond information for the ligand (see add. inf.). 
['id'] (str) The PDB code of the structure.
['seq'] (list) A list of all the sequences of each protien chain.
['smiles'] (str) The SMILES string of the ligand.
['scores'] (dict) All labels, currently only the negative log of the affinity. 
['types'] (dict) Type of each entry.
['file_path'] (str) Path to the LMDB.


## Additional Information

For some models, particularly GNNs, it is useful to use the original bonding information
for the small molecule ligands. Therefore, we provide this information in a separate
'bonds' dataframe. This dataframe links ensemble name (PDB ID) with a square matrix
representing the bonds between atoms in the molecule. Bond types are encoded as floats:
single (1.0). double (2.0), triple (3.0), and aromatic (1.5).

