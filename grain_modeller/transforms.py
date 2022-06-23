'''
Name:
    Transforms
Description:
    Contains methods which transform Atoms, Unitcell, and Supercells in a
reversible way, e.g. affine transformations.

'''
import numpy as np
import linear_algebra as linalg
from atom import Atom
from supercell import SuperCell
import copy


def rotate(structure, matrix):
    '''
    Rotates the atoms in the structured array of an Atom or Supercell using the
    provided rotation matrix. If the structure is an Atom it's rotated by
    changing its fractional coordinates, if it's a supercell it's rotated by
    changing its vector space.
    '''
    if isinstance(structure, Atom): rotate_atom(structure, matrix)
    if isinstance(structure, SuperCell): rotate_supercell(structure, matrix)


def rotate_atom(atom, matrix):
    '''
    Rotates an Atom object using the given matrix to alter its fractional
    coordinates.
    '''
    coordinates = atom.fractional['coordinates'].T
    coordinates = (matrix @ coordinates).T
    atom.fractional['coordinates'] = np.around(coordinates, 6)


def rotate_supercell(supercell, matrix):
    '''
    Rotates a SuperCell object using the given matrix to alter its vector
    space, and to rotate it's vectors.
    '''
    vector_space = copy.deepcopy(supercell.vector_space)
    vector_space = matrix @ vector_space
    supercell.a_side_vector = supercell.x_repeat*vector_space[:, 0]
    supercell.b_side_vector = supercell.y_repeat*vector_space[:, 1]
    supercell.c_side_vector = supercell.z_repeat*vector_space[:, 2]
    supercell.vector_space = vector_space
    if supercell.cartesian is not None: supercell.set_cartesian()


def translate(structure, vector, coordinates='fractional'):
    '''
    Translates the structured array via its fractional coordinates by default,
    however can translate in cartesian also.
    '''
    vector = np.array(vector)
    if coordinates == 'cartesian':
        inverse_vector_space = np.linalg.inv(structure.vector_space)
        vector = inverse_vector_space @ vector
    structure.fractional['coordinates'] += vector
    if not structure.cartesian is None: structure.set_cartesian()
