
python get_embeddings.py \
                --model_dir runs/hae-z=128-lmax=6-collected_lmax=6-rst_norm=square \
                --pdb_list test_pdbs.csv \
                --pdb_dir /gscratch/stf/mpun/data/casp12/pdbs/validation \
                --pdb_processing all_at_once \
                --batch_size 1000 \
                --output_filename hae_test
