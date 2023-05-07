import os
import yaml
import json
import argparse

from src.training import hvae_training, hvae_inference, hvae_standard_evaluation
from holographic_vae.utils.argparse import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True,
                        help='Custom name that identifies the model') # custom name that identifies the model


    parser.add_argument('--experiment_dir', type=str, default='runs',
                        help='Parent directory to all the experiments.')
    
    parser.add_argument('--training_config', type=optional_str, default='config.yaml',
                        help='Ignored when --eval_only is toggled.')
    
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()


    args.experiment_dir = os.path.join(args.experiment_dir, args.model_id)

    # make directory if it does not already exist
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    
    # load config if requested, if config is None, then use hparams within experiment_dir
    # if in eval_only mode, then load hparams from experiment_dir
    if args.training_config is not None and not args.eval_only:
        with open(args.config_file, 'r') as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)

        # save hparams as json file within expriment_dir
        with open(os.path.join(args.experiment_dir, 'hparams.json'), 'w+') as f:
            json.dump(hparams, f, indent=4)
        
    else:
        with open(os.path.join(args.experiment_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)

    if not args.eval_only:
        # launch training script
        hvae_training(args.experiment_dir)

    # perform inference, on standard test data, with basic results
    hvae_inference(args.experiment_dir, normalize_input_at_runtime=True, model_name='lowest_total_loss_with_final_kl_model')
    hvae_standard_evaluation(args.experiment_dir, model_name='lowest_total_loss_with_final_kl_model')
