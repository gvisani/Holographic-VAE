
## Downloading the ATOM3D LBA dataset

Download and unzip ATOM3D LBA data by running

```bash
bash download_and_untar_data.sh
```

The `atom3d` package contains utility functions that are useful for model training, and we rely on it for our pipeline. It can be simply installed via `pip`:

```bash
pip install atom3d
```

## Making predictions using a pre-trained H-(V)AE model

The script `atom3d_lba_prediction.py` can be used to predict Ligand Binding Affinity on the ATOM3D dataset, using a pretrained H-(V)AE model (saved, along with its `hparams.json` file, in `MODEL_DIR`).
The script first extracts H-(V)AE embeddings for residues in the pocket. Following ATOM3D, we consider residues to be in the pocket if their $C\alpha$ is within $6.0 \AA$ of any atom in the ligand.
; on a single A40 GPU and a model with ~2.6M parameters, this takes about 15 minutes. The embeddings are stored in `MODEL_DIR`, so that they only need to be computed once for each H-(V)AE model.

For example, to make predictions on the 30% similairty split, using an ensemble of 10 Random Forests trained on top of sums of embeddings our pretrained H-AE saved in `../protein_neighborhoods/runs/hae-z=128-lmax=6-collected_lmax=6-rst_norm=square/`, run

```bash
python atom3d_lba_prediction.py --model_dir ../protein_neighborhoods/runs/hae-z=128-lmax=6-collected_lmax=6-rst_norm=square/ \
                                --data_path ./data/ \
                                --pdb_similarity_split 30 \
                                --regressor RF \
                                --aggr_fn sum \
                                --num_bootstrap_samples 10
```

The script will automatically select hyperparameters that minimize the validation RMSD, over pre-defined grids of values. To see, and eventually modify, the grids, please refer to the function `select_best_hparams()` in `./src/training/kd_prediction.py`.

Note: to use gradient-boosted trees as regressor, install the `xgboost` package via:

```bash
pip install xgboost
```
