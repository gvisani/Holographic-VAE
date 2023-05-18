import h5py
from functools import partial
import numpy as np
from sklearn.neighbors import KDTree

from typing import *

NUM_FIELDS = 5

def get_neighborhoods(proteins: np.ndarray,
                      r: float = 10.0,
                      remove_central_residue: bool = True,
                      backbone_only: bool = False,
                      max_atoms: int = 1000
                      ) -> np.ndarray:
    '''
    Collect neighborhoods from preprocessed protein
    '''

    dt = np.dtype([
        ('res_id','S50', (6)),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S50', (max_atoms, 6)),
        ('coords', 'f4', (max_atoms, 3)),
        # ('SASAs', 'f4', (max_atoms)),
        # ('RSAs', 'f4', (max_atoms)),
        # ('charges', 'f4', (max_atoms)),
    ])
    
    neighborhoods = []
    num_nbs = 0
    for np_protein in proteins:
        nbs = get_padded_neighborhoods(np_protein, r=r, remove_central_residue=remove_central_residue, backbone_only=backbone_only, padded_length=max_atoms)
        neighborhoods.append(nbs)
        num_nbs += len(nbs)

    np_neighborhoods = np.zeros(shape=(num_nbs,), dtype=dt)
    i = 0
    for nbs in neighborhoods:
        for nb in nbs:
            np_neighborhoods[i] = (*nb,)
            i += 1

    return np_neighborhoods


################################################################################


BACKBONE_ATOMS = [b' N  ', b' CA ', b' C  ', b' O  ']
BACKBONE_ATOMS_PLUS_CB = [b' N  ', b' CA ', b' C  ', b' O  ', b' CB ']

# def remove_central_residue(nb):
#     central_res_id = nb['res_id']
#     res_ids = nb['res_ids']
#     mask = np.logical_and(res_ids != central_res_id, res_ids != b'')
#     return nb[mask]

# slice array along given indices
def slice_array(arr,inds):
    return arr[inds]

# given a set of neighbor coords, slice all info in the npProtein along neighbor inds
def get_neighbors(neighbor_inds, npProtein):
    return list(map(partial(slice_array, inds=neighbor_inds), npProtein))

def get_padded_neighborhoods(np_protein, r=10.0, remove_central_residue=True, backbone_only=False, padded_length=1000):
    neighborhoods = get_neighborhoods_from_protein(np_protein, r=r, remove_central_residue=remove_central_residue, backbone_only=backbone_only)
    padded_neighborhoods = pad_neighborhoods(neighborhoods, padded_length=padded_length)
    return padded_neighborhoods

def get_neighborhoods_from_protein(np_protein, r=10.0, remove_central_residue=True, backbone_only=False):
    atom_names = np_protein['atom_names']
    real_locs = (atom_names != b'')
    atom_names = atom_names[real_locs]
    coords = np_protein['coords'][real_locs]
    ca_locs = (atom_names == b' CA ')
    ca_coords = coords[ca_locs]
    ca_res_ids = np_protein['res_ids'][real_locs][ca_locs]
    tree = KDTree(coords, leaf_size=2)

    neighbors_list = tree.query_radius(ca_coords, r=r, count_only=False)
    
    get_neighbors_custom = partial(
        get_neighbors,                          
        npProtein = [np_protein[x] for x in range(1,NUM_FIELDS)]
    )

    res_ids = np_protein['res_ids'][real_locs]
    nh_ids = np_protein['res_ids'][real_locs][ca_locs]
    nh_atoms = np_protein['atom_names'][real_locs]

    # remove non-backbone atoms
    if backbone_only:
        for i, nh_id, neighbor_list in zip(np.arange(len(nh_ids)), nh_ids, neighbors_list):
            neighbors_list[i] = [x for x in neighbor_list if nh_atoms[x] in BACKBONE_ATOMS]
    
    # remove central residue atoms
    if remove_central_residue:
        for i, nh_id, neighbor_list in zip(np.arange(len(nh_ids)), nh_ids, neighbors_list):
            neighbors_list[i] = [x for x in neighbor_list if np.logical_or.reduce(res_ids[x] != nh_id, axis=-1)]
    else:
        # still must always remove central alpha carbon, otherwise we get nan values since its radius is 0.0
        for i, nh_id, neighbor_list in zip(np.arange(len(nh_ids)), nh_ids, neighbors_list):
            neighbors_list[i] = [x for x in neighbor_list if (np.logical_or.reduce(res_ids[x] != nh_id, axis=-1) or nh_atoms[x] != b' CA ')] # if (np.logical_or.reduce(res_ids[x] != nh_id, axis=-1) or nh_atoms[x] != b' CA ')
    
    neighborhoods = list(map(get_neighbors_custom, neighbors_list))
    
    for nh,nh_id,ca_coord in zip(neighborhoods,nh_ids,ca_coords):
        # center on alpha carbon
        nh[3] = np.array(nh[3] - ca_coord)

        nh.insert(0, nh_id)

    return neighborhoods

# given a matrix, pad it with empty array
def pad(arr,padded_length=100):
    try:
        # get dtype of input array
        dt = arr[0].dtype
    except IndexError as e:
        print(e)
        print(arr)
        raise Exception
    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.empty(padded_shape, dtype=dt)
    
    # if type is string fill array with empty strings
    if np.issubdtype(bytes, dt):
        mat_arr.fill(b'')

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_neighborhood(
    ragged_structure,
    padded_length=100
):
    
    
    pad_custom = partial(pad,padded_length=padded_length)
    
    mat_structure = list(map(pad_custom,ragged_structure))

    return mat_structure

def pad_neighborhoods(
        neighborhoods,
        padded_length=600
):
    padded_neighborhoods = []
    for i,neighborhood in enumerate(neighborhoods):
        padded_neighborhoods.append(
            pad_neighborhood(
                [neighborhood[i] for i in range(1,NUM_FIELDS)],
                padded_length=padded_length
            )
        )
    [padded_neighborhood.insert(0,nh[0]) for nh,padded_neighborhood in zip(neighborhoods,padded_neighborhoods)]
    return padded_neighborhoods
