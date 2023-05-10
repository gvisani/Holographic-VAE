
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from time import time

from experiments.protein_neighborhoods.src.training import hvae_inference

import argparse
from holographic_vae.utils.argparse import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model.')

    parser.add_argument('--pdb_list', type=str, required=True,
                        help='csv file containing list of PDB files of interest, under the column "pdb".')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory containing PDB files.')
    
    parser.add_argument('--output_filename', type=str, default='./hae',
                        help='Output file name.')
    
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Will only make a difference if running inference on multiple PDBs at once.')

    parser.add_argument('--pdb_processing', type=str, default='all_at_once',
                        choices = ['all_at_once', 'one_at_a_time'],
                        help='Whether to process all pdbs at once before running inference, or perform inference one pdb at a time.\
                             "all_at_once" is faster and can benefit from batching, but requires more memory.\
                             "one_at_a_time" is slower and cannot benefit from batching, but requires less memory.')
    
    args = parser.parse_args()

    if args.pdb_processing == 'one_at_a_time':
        all_res_ids, all_inv_embeddings, all_learned_frames = [], [], []
        all_cosine_loss = []
        start = time()
        for pdb in tqdm(pd.read_csv(args.pdb_list)['pdb']):
            pdb_file = pdb + '.pdb'
            if pdb_file not in os.listdir(args.pdb_dir):
                print('Skipping {} because it is not in the PDB directory.'.format(pdb_file))
                continue

            inv_embeddings, learned_frames, _, res_ids, _, _, cosine_loss = hvae_inference(args.model_dir,
                                                                                            output_filepath=None, # ensures that the embeddings are not saved to disk, but instead returned
                                                                                            data_filepath=os.path.join(args.pdb_dir, pdb_file),
                                                                                            verbose=False,
                                                                                            loading_bar=False,
                                                                                            batch_size=args.batch_size)

            all_res_ids.append(res_ids)
            all_inv_embeddings.append(inv_embeddings)
            all_learned_frames.append(learned_frames)
            all_cosine_loss.append(cosine_loss)
        
        print('Took {} seconds to generate embeddings for {} PDBs.'.format(time() - start, pd.read_csv(args.pdb_list)['pdb'].shape[0]))
        
        all_res_ids = np.hstack(all_res_ids)
        all_inv_embeddings = np.vstack(all_inv_embeddings)
        all_learned_frames = np.vstack(all_learned_frames)
        print('Cosine loss: {} (+- {} across PDBs)'.format(np.mean(all_cosine_loss), np.std(all_cosine_loss)))

    elif args.pdb_processing == 'all_at_once':
        start = time()
        all_inv_embeddings, all_learned_frames, _, all_res_ids, _, _, all_cosine_loss = hvae_inference(args.model_dir,
                                                                                                       data_filepath=args.pdb_list,
                                                                                                       pdb_dir=args.pdb_dir,
                                                                                                       loading_bar=True,
                                                                                                       batch_size=args.batch_size)
        
        print('Took {} seconds to generate embeddings for {} PDBs.'.format(time() - start, pd.read_csv(args.pdb_list)['pdb'].shape[0]))

        print('Cosine loss: {}'.format(all_cosine_loss))

    
    ## save res_ids to csv file so they are easily indexable
    ## save embeddings and frames in a typed numpy array of floats
    
    all_res_ids = np.vstack([np.array(res_id.split('_')) for res_id in all_res_ids])
    columns = np.array(['residue', 'pdb', 'chain', 'resnum', 'insertion_code', 'secondary_structure'])
    all_res_ids = np.vstack([columns.reshape(1, -1), all_res_ids])
    all_res_ids = all_res_ids[:, np.array([1, 2, 0, 3, 4])] # rearrange to put pdb in front, and remove secondary struucture
    columns = all_res_ids[0]
    all_res_ids = all_res_ids[1:]

    # save res ids in a csv file and embeddings and frames in a numpy array
    pd.DataFrame(all_res_ids, columns=columns).to_csv(args.output_filename + '-res_ids.csv', index=False)
    np.savez(args.output_filename + '-embeddings_and_frames.npz',
                        embeddings=all_inv_embeddings,
                        frames=all_learned_frames)
    