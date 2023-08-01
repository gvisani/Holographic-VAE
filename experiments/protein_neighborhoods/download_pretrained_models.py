""" Script to download pretrained models """

import os
import urllib.request

MODEL_ID_TO_URL = {
    'HAE-z=128-lmax=6-rst_norm=square': "https://figshare.com/ndownloader/files/40710101",
    'HAE-z=256-lmax=6-ls_rule=full-rst_norm=square': "https://figshare.com/ndownloader/files/41791344"
}

def download_weights(
    model_id,
    download_dir = "./runs/",
):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    tar_file_path = os.path.join(download_dir, model_id + '.tar.gz')
    try:
        urllib.request.urlretrieve(
            MODEL_ID_TO_URL[model_id],
            tar_file_path
        )
        print("Model successfully downloaded to "\
              f"{tar_file_path}")
    except Exception as e:
        print(e)
        github = "https://github.com/gvisani/Holographic-VAE"
        print("Model weights could not be downloaded\n" \
              f"Please see {github} for help")

    os.system(
        f"tar -xvzf {tar_file_path} -C {download_dir}"      
    )
    os.system(f"rm {tar_file_path}")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='HAE-z=128-lmax=6-rst_norm=square',
                                                choices=['HAE-z=128-lmax=6-rst_norm=square', 'HAE-z=256-lmax=6-ls_rule=full-rst_norm=square'],
                                                help='Model ID to download.')
    parser.add_argument('--download_dir', type=str, default='./runs/',
                        help='Directory to download the model to. Set to ./runs/ by default.')
    args = parser.parse_args()

    download_weights(args.model_id, args.download_dir)