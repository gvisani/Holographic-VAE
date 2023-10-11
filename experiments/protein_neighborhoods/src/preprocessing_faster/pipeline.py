
from protein_holography_pytorch.preprocessing_faster import get_structural_info, get_structural_info_from_pyrosetta_pose, get_neighborhoods, get_zernikegrams

def get_zernikegrams_from_structures(pdb_or_pdblist_or_pose,
                                     get_neighborhoods_kwargs,
                                     get_zernikegrams_kwargs,
                                     max_atoms=20000):

    if isinstance(pdb_or_pdblist_or_pose, str) or isinstance(pdb_or_pdblist_or_pose, list):
        proteins = get_structural_info(pdb_or_pdblist_or_pose, max_atoms)
    else: # pyrosetta pose
        proteins = get_structural_info_from_pyrosetta_pose(pdb_or_pdblist_or_pose, max_atoms)
    
    neighborhoods = get_neighborhoods(proteins, **get_neighborhoods_kwargs)

    zernikegrams = get_zernikegrams(neighborhoods, **get_zernikegrams_kwargs)

    return zernikegrams



