
import numpy as np

'''
Notes:
    - Hetero atoms identified with H_CA (Calcium ion?) has name and fullname equal to CA, which is the same as alpha carbons!
      So we need to use other information to dinstinguish between the two cases. Currently, also using element == C
    
'''


def remove_hydrogens(item, atoms_key):
    '''
    :param item: Dataset item to transform
    :type item: dict
    :param atoms_key: key in item pointing to a dataframe of atoms, from which to remove hydrogens
    :type atom_key: str

    :return: Dataset item atoms_key now points to the same dataframe but with hydogens removed
    '''

    item[atoms_key] = item[atoms_key].loc[item[atoms_key]['element'] != 'H']

    return item


def get_pocket_at_radius(item, radius=6.0):
    raise NotImplementedError('get_pocket_at_radius not implemented yet')

def remove_water(item, atoms_key):
    '''
    :param item: Dataset item to transform
    :type item: dict
    :param atoms_key: key in item pointing to a dataframe of atoms, from which to remove waters
    :type atom_key: str

    :return: Dataset item atoms_key now points to the same dataframe but with waters removed
    '''

    item[atoms_key] = item[atoms_key].loc[item[atoms_key]['hetero'] != 'W']

    return item

def remove_noncanonical_insertion_codes(item, atoms_key):
    '''
    :param item: Dataset item to transform
    :type item: dict
    :param atoms_key: key in item pointing to a dataframe of atoms, from which to remove waters
    :type atom_key: str

    :return: Dataset item atoms_key now points to the same dataframe but with noncanonical (i.e. != ' ') removed

    NB: the residues with non-canonical insertion codes are usually NOT at the same location as their canonical counterpart! They are in the vicinity, but not quite there. I wanna see them in pymol.
    '''

    item[atoms_key] = item[atoms_key].loc[item[atoms_key]['insertion_code'] == ' ']

    return item

def remove_hetero(item, atoms_key, remove_water=True):
    '''
    By default, also removes water molecules, but that can be changed by setting remove_water=False
    '''

    if remove_water:
        mask = np.logical_or(item[atoms_key]['hetero'] == ' ', item[atoms_key]['hetero'] == 'W')
    else:
        mask = item[atoms_key]['hetero'] == ' '

    item[atoms_key] = item[atoms_key].loc[mask]

    return item

def standardize_nonprotein_elements(item, atoms_key):
    '''
    Standardize non-protein elements to X

    Currently returns the following warning:
        /mmfs1/gscratch/spe/gvisan01/Holographic-VAE/experiments/ATOM3D_LBA/src/utils/data/functional.py:80: SettingWithCopyWarning: 
        A value is trying to be set on a copy of a slice from a DataFrame

        See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        item[atoms_key]['element'][mask] = 'X'
    '''
    mask = np.logical_and.reduce([item[atoms_key]['element'] != 'C', item[atoms_key]['element'] != 'N', item[atoms_key]['element'] != 'O', item[atoms_key]['element'] != 'S', item[atoms_key]['element'] != 'H'])

    item[atoms_key]['element'][mask] = 'X'

    return item


        