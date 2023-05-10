
## Pipeline for constructing neighborhood-level zernike projections (zernikegrams) from a list of PDBs

**Note:** *We will remove the dependency on pyrosetta in the future.*

The pipeline is divided in three steps, each with its corresponding script:
1. `get_structural_info_pipeline.py` takes a list of PDBs as input and uses pyrosetta to parse them, saving structural information (atom coordinates, elements, etc.) into an hdf5 file. Make sure your local version of pyrosetta is appended to the system's PATH in line 13 of `get_structural_info.py`.
2. `get_neighborhoods_pipeline.py` takes the hdf5 file generated in step 1 as input and constructs neighborhoods around each atom in the PDBs. The neighborhoods are saved into a new hdf5 file.
3. `get_zernikegrams_pipeline.py` takes the hdf5 file generated in step 2 as input and constructs zernikegrams for each neighborhood. The zernikegrams are saved into a new hdf5 file.

For seamless compatibility with our training and evaluation scripts, we recommend processing data splits independently, and having the output zernikegram files follow the following pattern: `{USER_DEFINED_NAME}_{split}-lmax=%d-r=%.1f-rst_normalization=%s.hdf5`

Below are the help messages for each script:

```
usage: get_structural_info_pipeline.py [-h] --output_hdf5 OUTPUT_HDF5 [--output_dataset_name OUTPUT_DATASET_NAME] [--parallelism PARALLELISM] --pdb_list PDB_LIST --pdb_dir PDB_DIR

optional arguments:
  -h, --help            show this help message and exit
  --output_hdf5 OUTPUT_HDF5
                        User-defined name of output hdf5 file that will contain the per-pdb structural information.
  --output_dataset_name OUTPUT_DATASET_NAME
                        Name of the dataset within output_hdf5 where the structural information will be saved. We recommend keeping this set to simply "data".
  --parallelism PARALLELISM
                        Parallelism for multiprocessing.
  --pdb_list PDB_LIST   csv file containing list of PDB files of interest, under the column "pdb".
  --pdb_dir PDB_DIR     Directory containing PDB files.
```

```
usage: get_neighborhoods_pipeline.py [-h] --input_hdf5 INPUT_HDF5 --output_hdf5 OUTPUT_HDF5 [--input_dataset_name INPUT_DATASET_NAME] [--output_dataset_name OUTPUT_DATASET_NAME] [--parallelism PARALLELISM] [--radius RADIUS] [--remove_central_residue REMOVE_CENTRAL_RESIDUE] [--backbone_only BACKBONE_ONLY] [--filter_out_chains_not_in_proteinnet FILTER_OUT_CHAINS_NOT_IN_PROTEINNET] [--AAs AAS]

optional arguments:
  -h, --help            show this help message and exit
  --input_hdf5 INPUT_HDF5
                        Path to hdf5 file containing protein 3D structure information. Must be output to the script `get_structural_info_pipeline.py`
  --output_hdf5 OUTPUT_HDF5
                        User-defined name of output hdf5 file that will contain the extracted neighborhoods.
  --input_dataset_name INPUT_DATASET_NAME
                        Name of the dataset within input_hdf5 where the structural information is to be found. We recommend keeping this set to simply "data".
  --output_dataset_name OUTPUT_DATASET_NAME
                        Name of the dataset within output_hdf5 where the neighborhoods information will be saved. We recommend keeping this set to simply "data".
  --parallelism PARALLELISM
                        Parallelism count for multiprocessing
  --radius RADIUS       Radius of the neighborhoods. Alias of "rcut".
  --remove_central_residue REMOVE_CENTRAL_RESIDUE
                        Whether to remove the central residue. Set it to False for H-(V)AE neighborhoods.
  --backbone_only BACKBONE_ONLY
                        Whether to keep only backbone atoms. Set it to False for H-(V)AE neighborhoods.
  --filter_out_chains_not_in_proteinnet FILTER_OUT_CHAINS_NOT_IN_PROTEINNET
                        Whether to exclude neighborhoods that do not appear in casp12 proteinnet, as only some chains within the provided PDBs are in proteinnet. Set to True in our experiments on CASP 12, but it
                        requires access to the proteinnet data.
  --AAs AAS             List of amino-acid types to collect. Either "all" or provided in comma-separated form.
```

```
usage: get_zernikegrams_pipeline.py [-h] --input_hdf5 INPUT_HDF5 --output_hdf5 OUTPUT_HDF5 [--input_dataset_name INPUT_DATASET_NAME] [--output_dataset_name OUTPUT_DATASET_NAME] [--parallelism PARALLELISM] [--rmax RMAX] [--rcut RCUT] [--lmax LMAX] [--backbone_only BACKBONE_ONLY] [--request_frame REQUEST_FRAME] [--rst_normalization {None,None,square}] [--channels CHANNELS] [--get_psysicochemical_info_for_hydrogens GET_PSYSICOCHEMICAL_INFO_FOR_HYDROGENS]

optional arguments:
  -h, --help            show this help message and exit
  --input_hdf5 INPUT_HDF5
                        Path to hdf5 file containing collected protein neighborhoods. Must be output to the script `get_neighborhoods_pipeline.py`
  --output_hdf5 OUTPUT_HDF5
                        User-defined name of output hdf5 file that will contain the zernikegrams.
  --input_dataset_name INPUT_DATASET_NAME
                        Name of the dataset within input_hdf5 where the neighborhoods information is to be found. We recommend keeping this set to simply "data".
  --output_dataset_name OUTPUT_DATASET_NAME
                        Name of the dataset within output_hdf5 where the zernikegram information will be saved. We recommend keeping this set to simply "data".
  --parallelism PARALLELISM
                        Parallelism count for multiprocessing. Keep this set to 1, as the current version runs **slower** with multiprocessing.
  --rmax RMAX           Maximum radial order.
  --rcut RCUT           Radius of the neighborhoods. Alias of "radius".
  --lmax LMAX           Maximum spherical order.
  --backbone_only BACKBONE_ONLY
                        Whether to keep only backbone atoms. Set it to False for H-(V)AE neighborhoods.
  --request_frame REQUEST_FRAME
                        Whether to request the backbone frame. Unused in our experiments.
  --rst_normalization {None,None,square}
                        Normalization type per Dirac-Delta projection.
  --channels CHANNELS   Atomic and physicochemical channels to be included in the Zernikegrams. Can be any combination of [C, N, O, S, H, SASA, charge].
  --get_psysicochemical_info_for_hydrogens GET_PSYSICOCHEMICAL_INFO_FOR_HYDROGENS
                        Whether to include physicochemical information for hydrogens. Only applies if requesting SASA or charge.
```
