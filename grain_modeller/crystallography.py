'''
Name:
    Crystallography
Description:
    Contains functions specific to crystallographic manipulation, e.g.
    producing plane intercepts from a (hkl) plane definition.
'''
import numpy as np
import linear_algebra as linalg
from scipy import spatial
import cartesian_edits as ce


def miller_to_intercepts(plane):
    '''
    Takes in a plane defined using (hkl) miller indices. Inverts these to get
    the intercepts of the plane along the crystallographic axes. For example,
    an intercept of a half would be half way along the vector defining its
    crystallographic axis.

    For more information on Miller indices please see:
    https://en.wikipedia.org/wiki/Miller_index
    '''
    intercepts = np.array(plane).astype(float)
    if intercepts[0] != 0:
        intercepts[0] = 1/intercepts[0]
    else:
        intercepts[0] = np.inf
    if intercepts[1] != 0:
        intercepts[1] = 1/intercepts[1]
    else:
        intercepts[1] = np.inf
    if intercepts[2] != 0:
        intercepts[2] = 1/intercepts[2]
    else:
        intercepts[2] = np.inf
    return intercepts


def cartesian_plane_normal(vector_space, plane):
    '''
    Using a vector_space and plane given in (hkl) miller indices calculate the
    associated plane normal in cartesian coordinates. Plane normal is a unit
    vector pointing away from the origin.

    Providing a plane normal in this absolute format is necessary for making
    cuts in a common vector space that is independent of a supercell's vector
    space.
    '''
    intercepts = miller_to_intercepts(plane)
    normal = linalg.plane_normal(vector_space, intercepts)
    return normal


def distance_symmetries(atoms, neighbours=12, box=None):
    '''
    Finds the distance symmetries present in a crystal between each atom and
    its nearest neighbours. The number of neighbours considered is alterable;
    the atoms considered can be defined via box selection.

    All atoms have their distance symmetries calculated, but only atoms within
    the box will have those symmetries returned. This allows you to narrow down
    the atom symmetries to bulk or surface for example.

    atoms: Numpy array of atom coordinates (generally in cartesian).
    neighbours: Number of nearest neighbours to search for each atom.
    box: Numpy array, list, or tuple defining the minimum and maximum edges of
        a box in 3D. Defined as a row array of the form:
                            [[x_min, x_max],
                             [y_min, y_max],
                             [z_min, z_max]]
    '''
    distances = neighbour_distances(atoms, neighbours)
    if box is not None:
        box = np.array(box)
        box_mask = ce.box_select(atoms, box)
        distances = distances[box_mask]
    distances = np.unique(distances, axis=0)
    return distances


def neighbour_distances(atoms, neighbours=12):
    '''
    Calculates the distances between a numpy array of atoms and their closest
    neighbours. The number of neighbours searched is defined by the neighbours
    variable, set to 12 as this is maximum possible number of nearest
    neighbours in a crystal.
    '''
    atom_tree = spatial.cKDTree(atoms)
    distances, indexes = atom_tree.query(atoms, k=neighbours+1)
    distances = np.around(distances, 6)
    return distances
