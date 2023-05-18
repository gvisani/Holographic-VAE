
## Pipeline for constructing neighborhood-level zernike projections (zernikegrams) from a list of PDBs

The pipeline is divided in three steps, each with its corresponding script:
1. `get_structural_info_pipeline.py` takes a list of PDBs as input and uses pyrosetta to parse them, saving structural information (atom coordinates, elements, etc.) into an hdf5 file.
2. `get_neighborhoods_pipeline.py` takes the hdf5 file generated in step 1 as input and constructs neighborhoods around each atom in the PDBs. The neighborhoods are saved into a new hdf5 file.
3. `get_zernikegrams_pipeline.py` takes the hdf5 file generated in step 2 as input and constructs zernikegrams for each neighborhood. The zernikegrams are saved into a new hdf5 file.

For seamless compatibility with our training and evaluation scripts, we recommend processing data splits independently, and having the output zernikegram files follow the following pattern: `{USER_DEFINED_NAME}_{split}-lmax=%d-r=%.1f-rst_normalization=%s.hdf5`

**Note:** The use of pyrosetta is not strictly necessary. Since our H-(V)AE models do not use pyrosetta-computed physicochemical quantities (SASA and charge, which are computed within our pipeline but not used), other parsers can be used instead, and be made compatible with our pipeline by following the output format of `get_structural_info_pipeline.py`.
