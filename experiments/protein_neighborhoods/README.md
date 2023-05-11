
## On the current necessity of pyrosetta

Currently, our code is dependent on pyrosetta which can be downloaded from [here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d). A license is available at no cost to academics and can be obtained [here](https://www.pyrosetta.org/home/licensing-pyrosetta). However, the use of pyrosetta can be bypassed by users if they write a custom script that parses PDB files; see the README in the `preprocessing` folder for more details. We plan on releasing a script using an alternative PDB parser soon.


## Extracting H-(V)AE embeddings from a series of PDBs

*Automatic download of pre-trained models is not yet supported. Send us an email to get access to the pre-trained models from the paper.*

Use the `get_embeddings.py` script to get embeddings from a pre-trained model.
The script takes as input a series of pdbs, and outputs invariant (L = 0) embeddings and equivariant (L = 1) frames.
The script can be run as follows:

```
usage: get_embeddings.py [-h] --model_dir MODEL_DIR --pdb_list PDB_LIST --pdb_dir PDB_DIR [--output_filename OUTPUT_FILENAME] [--batch_size BATCH_SIZE] [--pdb_processing {all_at_once,one_at_a_time}]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory containing the trained model.
  --pdb_list PDB_LIST   csv file containing list of PDB files of interest.
  --pdb_dir PDB_DIR     Directory containing PDB files.
  --output_filename OUTPUT_FILENAME
                        Output file name.
  --batch_size BATCH_SIZE
                        Will only make a difference if running inference on multiple PDBs at once.
  --pdb_processing {all_at_once,one_at_a_time}
                        Whether to process all pdbs at once before running inference, or perform inference one pdb at a time. "all_at_once" is faster
                        and can benefit from batching, but requires more memory. "one_at_a_time" is slower and cannot benefit from batching, but
                        requires less memory.
```

## Training a new model

**Step 1:** Pre-process PDBs using the preprocessing pipeline found in the `preprocessing` folder.
The pipeline will output three hdf5 files (train/valid/test) containing zernike projections of
the protein neighborhoods in the PDBs.

**Step 2:** Train the model using the `train_and_eval.py` script. The script can be run as follows:

```
    usage: train_and_eval.py [-h] --model_dir MODEL_DIR [--training_config TRAINING_CONFIG] [--eval_only]

    optional arguments:
    -h, --help            show this help message and exit
    --model_dir MODEL_DIR
                            Directory containing the model and experimental results related to it.
    --training_config TRAINING_CONFIG
                            Ignored when --eval_only is toggled.
    --eval_only           If toggled, then only perform inference and evaluation on the standard test data.
```

