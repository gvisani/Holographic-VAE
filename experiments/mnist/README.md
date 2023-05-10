
## Holographic-(V)AE on MNIST-on-the-Sphere

**Step 1:** Generate data using `bash generate_datasets.sh`. This script downloads the MNIST datasets, projects it onto a sphere, and performs a forward fourier projection into spherical hamonics space. By default, the script processes the NR/NR, NR/R, and R/R datasets ({train}/{test} ; NR: Not-Rotated ; R: Rotated)

**Step 2:** Use the `train_and_eval.py` script to train and evaluate the model, alongside the confration in `config.yaml`. The script leverages functions for training and inference found in `./src/training`, which can be used as templates to train and evaluate H-(V)AE on new spherical image datasets.

For an example training run, using the provided config file, run the followinig commands:
```
    bash generate_datasets.sh
    python train_and_eval.py --model_dir runs/hvae-z=16-NRR --training_config config.yaml
```
