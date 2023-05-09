
## Extracting H-(V)AE embeddings from a serries of PDBs

Use the `get_embeddings.py` script to get embeddings from a pre-trained model. The script takes as input a series of pdbs,
and outputs invariant (L = 0) embeddings and equivariant (L = 1) frames. The script can be run as follows:

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
