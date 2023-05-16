
import os

import json

def get_pdb_to_neglogkdki_by_split(path_to_raw_data, path_to_indices):

    if os.path.exists(os.path.join(path_to_indices, 'train_pdb_to_neglogkdki.json')) and os.path.exists(os.path.join(path_to_indices, 'val_pdb_to_neglogkdki.json')) and os.path.exists(os.path.join(path_to_indices, 'test_pdb_to_neglogkdki.json')):
        print('Loading pdb_to_neglogkdki by split from file...')
        print()

        with open(os.path.join(path_to_indices, 'train_pdb_to_neglogkdki.json'), 'r') as f:
            train_pdb_to_neglogkdki = json.load(f)
        
        with open(os.path.join(path_to_indices, 'val_pdb_to_neglogkdki.json'), 'r') as f:
            val_pdb_to_neglogkdki = json.load(f)
        
        with open(os.path.join(path_to_indices, 'test_pdb_to_neglogkdki.json'), 'r') as f:
            test_pdb_to_neglogkdki = json.load(f)

        pdb_to_neglogkdki_by_split = {'train': train_pdb_to_neglogkdki, 'val': val_pdb_to_neglogkdki, 'test': test_pdb_to_neglogkdki}

        return pdb_to_neglogkdki_by_split

    print('Getting pdb_to_neglogkdki by split...')
    print()

    from atom3d.datasets import LMDBDataset

    dataset = LMDBDataset(path_to_raw_data)
    pdbs = [item['id'] for item in dataset] # Get all PDBs
    labels = [item['scores']['neglog_aff'] for item in dataset] # Get all labels

    # Get indices of PDBs in each split, save as json file
    pdb_to_neglogkdki_by_split = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(path_to_indices, f'{split}_indices.txt'), 'r') as f:
            indices = [int(line.strip()) for line in f.readlines()]
        pdbs_split = [pdbs[i] for i in indices]
        labels_split = [labels[i] for i in indices]
        pdb_to_neglogkdki = dict(zip(pdbs_split, labels_split))
        with open(os.path.join(path_to_indices, f'{split}_pdb_to_neglogkdki.json'), 'w') as f:
            json.dump(pdb_to_neglogkdki, f, indent = 4)
        
        pdb_to_neglogkdki_by_split[split] = pdb_to_neglogkdki
    
    return pdb_to_neglogkdki_by_split