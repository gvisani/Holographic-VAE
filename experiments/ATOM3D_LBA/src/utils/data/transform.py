
import numpy as np
import pandas as pd

from experiments.ATOM3D_LBA.src.utils.data import remove_hydrogens, remove_water, remove_noncanonical_insertion_codes, remove_hetero, standardize_nonprotein_elements

from typing import *


def neighborhoods_collate_fn(batch: List):
    coords, elements, pdbs = zip(*batch)
    return list(coords), list(elements), list(pdbs)


class NeighborhoodsTransform(object):

    def __init__(self, elements=['C', 'N', 'O', 'S'], **kwargs):
        '''
        Builds on top of BaseNeighborhoodsTransform, adding the following:
            - element filtering
            - converting to torch tensors

        '''

        self.elements = elements
        self.base_transform = BaseNeighborhoodsTransform(**kwargs)
    
    def __call__(self, item):

        import torch

        neighborhoods, _ = self.base_transform(item)

        x_coords_B_N3, x_elements_B_N3 = [], []

        for coords, elements in zip(neighborhoods['coords'], neighborhoods['elements']):
            mask = np.isin(elements, self.elements)
            x_coords_B_N3.append(torch.tensor(coords[mask], dtype=torch.float32))
            x_elements_B_N3.append(elements[mask].tolist())
        
        return x_coords_B_N3, x_elements_B_N3, item['id']
        
        



class BaseNeighborhoodsTransform(object):
    '''
    Given ATOM3DLBA data item, returns atomic pocket neighborhoods in parallel lists.
    To be used as a transform within ATOM3DLBA dataset class, for HCNN, HVAE and HGNN models

    Takes as input an ATOM3D LBA dataset item,
    then joins protein and ligand atoms into a dataframe,
    then extracts neighborhoods at all the CAs.
    Returns a list of neighborhoods

    the pocket rasidues are defined as the CAs present in item['atoms_pocket'].
    Then, a neighborhood for a particular CA is extracted as all atoms in pd.concat([item['atoms_protein'], item['atoms_ligand']]) within nb_radius of the CA

    By default, use the pocket provided by ATOM3D, which uses a cutoff of 6.0\AA. However, a different cutoff could be used.

    If remove_hetero is true, then no non-protein atoms should be there, so we can just ignore them. Otherwise, we can add a "all other elements" channel to the neighborhood, termed X, using the standardize_nonprotein_elements flag.

    NB: the residues with non-canonical insertion codes are usually NOT at the same location as their canonical counterpart! They are in the vicinity, but not quite there. I wanna see them in pymol.
    '''

    def __init__(self, nb_radius=10.0, remove_H=True, remove_water=True, remove_hetero=True, remove_noncanonical_insertion_codes=False, standardize_nonprotein_elements=True):
        self.nb_radius = nb_radius # called "rcut" in other functions/classed
        self.remove_H = remove_H
        self.remove_water = remove_water
        self.remove_hetero = remove_hetero
        self.remove_noncanonical_insertion_codes = remove_noncanonical_insertion_codes
        self.standardize_nonprotein_elements = standardize_nonprotein_elements

    def __call__(self, item):

        if self.remove_H:
            item = remove_hydrogens(item, 'atoms_protein')
            item = remove_hydrogens(item, 'atoms_pocket')
            item = remove_hydrogens(item, 'atoms_ligand')
        
        if self.remove_hetero:
            item = remove_hetero(item, 'atoms_protein', remove_water=self.remove_water)
            item = remove_hetero(item, 'atoms_pocket', remove_water=self.remove_water)
            item = remove_hetero(item, 'atoms_ligand', remove_water=self.remove_water)
        elif self.remove_water:
            item = remove_water(item, 'atoms_protein')
            item = remove_water(item, 'atoms_pocket')
            item = remove_water(item, 'atoms_ligand')
        
        if self.remove_noncanonical_insertion_codes:
            item = remove_noncanonical_insertion_codes(item, 'atoms_protein')
            item = remove_noncanonical_insertion_codes(item, 'atoms_pocket')
            item = remove_noncanonical_insertion_codes(item, 'atoms_ligand')
        
        if self.standardize_nonprotein_elements:
            item = standardize_nonprotein_elements(item, 'atoms_protein')
            item = standardize_nonprotein_elements(item, 'atoms_pocket')
            item = standardize_nonprotein_elements(item, 'atoms_ligand')
        
        item['atoms_protein'] = item['atoms_protein'].reset_index(drop=True)
        item['atoms_pocket'] = item['atoms_pocket'].reset_index(drop=True)
        item['atoms_ligand'] = item['atoms_ligand'].reset_index(drop=True)

        # join protein and ligand atoms into a single dataframe
        protein_ligand_joined_df = pd.concat([item['atoms_protein'], item['atoms_ligand']], ignore_index=True)

        # get alpha-carbons of pocket
        alpha_carbons_pocket_df = item['atoms_pocket'].loc[np.logical_and(item['atoms_pocket']['name'] == 'CA', item['atoms_pocket']['element'] == 'C')]

        # remove atoms (i.e. rows) in joined df of pocket CAs
        res_ids_ca = np.array(list(map(lambda x : '_'.join(map(str, x)), alpha_carbons_pocket_df[['resname', 'residue', 'chain', 'hetero', 'insertion_code', 'name']].values)))
        res_ids_all = np.array(list(map(lambda x : '_'.join(map(str, x)), protein_ligand_joined_df[['resname', 'residue', 'chain', 'hetero', 'insertion_code', 'name']].values)))
        locations_in_all = np.where(np.in1d(res_ids_all, res_ids_ca))[0]
        protein_ligand_joined_df = protein_ligand_joined_df.drop(locations_in_all).reset_index(drop=True)

        from scipy.spatial import KDTree
        tree_ca = KDTree(alpha_carbons_pocket_df[['x', 'y', 'z']].values)
        tree_all = KDTree(protein_ligand_joined_df[['x', 'y', 'z']].values)

        ca_to_all_neighbors_idxs = tree_ca.query_ball_tree(tree_all, r=self.nb_radius)

        # now, collect neighborhoods!
        coords = []
        elements = []
        atom_names = []
        res_ids = []
        for res_idx, res_nb_idxs in enumerate(ca_to_all_neighbors_idxs):
            coords.append(protein_ligand_joined_df[['x', 'y', 'z']].values[res_nb_idxs] - alpha_carbons_pocket_df[['x', 'y', 'z']].values[res_idx]) # center at the CA!
            elements.append(protein_ligand_joined_df['element'].values[res_nb_idxs])
            atom_names.append(protein_ligand_joined_df['name'].values[res_nb_idxs])
            res_ids.append(alpha_carbons_pocket_df[['resname', 'residue', 'chain', 'hetero', 'insertion_code']].values[res_idx])

        neighborhoods = {
            'res_ids' : res_ids,
            'coords' : coords,
            'elements' : elements,
            'atom_names' : atom_names,
        }

        return neighborhoods, item['id']
    
