import os
import yaml
import json
import argparse

from src.training import hvae_training, hvae_inference, hvae_standard_evaluation, classification_and_clustering_in_latent_space, hvae_reconstruction_tests
from src.utils.generate_dataset_of_conditional_samples import generate_dataset_of_conditional_samples
from holographic_vae.utils.argparse import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the model and experimental results related to it.') # custom name that identifies the model
    
    parser.add_argument('--training_config', type=optional_str, default='config.yaml',
                        help='Ignored when --eval_only is toggled.')
    
    parser.add_argument('--eval_only', action='store_true',
                        help='If toggled, then only perform inference and evaluation on the standard test data.')

    args = parser.parse_args()


    # make directory if it does not already exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # load config if requested, if config is None, then use hparams within model_dir
    # if in eval_only mode, then load hparams from model_dir
    if args.training_config is not None and not args.eval_only:
        with open(args.training_config, 'r') as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)

        # save hparams as json file within expriment_dir
        with open(os.path.join(args.model_dir, 'hparams.json'), 'w+') as f:
            json.dump(hparams, f, indent=4)
        
    else:
        with open(os.path.join(args.model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)

    if not args.eval_only:
        # launch training script
        hvae_training(args.model_dir)

    # perform inference, on standard test data, with basic results
    hvae_inference(args.model_dir, split='test', model_name='lowest_total_loss_with_final_kl_model', verbose=True, loading_bar=True)
    hvae_standard_evaluation(args.model_dir, split='test', model_name='lowest_total_loss_with_final_kl_model')
    classification_and_clustering_in_latent_space(args.model_dir, model_name='lowest_total_loss_with_final_kl_model', verbose=True, loading_bar=True)
    hvae_reconstruction_tests(args.model_dir, split='test', n_samples=5, model_name='lowest_total_loss_with_final_kl_model', verbose=True)
    generate_dataset_of_conditional_samples(args.model_dir)
    
