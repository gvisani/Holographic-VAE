import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='holographic_vae',
    version='0.0',
    author='Gian Marco Visani',
    author_email='gvisan01@cs.washington.edu',
    description='SO(3)-equivariant (variational) autoencoder for spherical images and atom-centered atomic environments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gvisani/holographic_vae',
    python_requires='>=3.8',
    install_requires='',
    packages=setuptools.find_packages(),
)

# os.chdir("holographic_vae/utils")
# os.system("python download_model_weights.py")
# os.chdir("../..")