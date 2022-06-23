'''
Name:
    Supercell
Description:
    Contains the supercell class, which defines a custom data object
    representing a supercell.
'''

import numpy as np
from scipy.linalg import norm


class SuperCell():

    def __init__(self, unit_cell, x_repeat, y_repeat, z_repeat):
        '''
        Instantiate a new supercell, with a unit_cell as a basis, and a number
        of repeats in the x, y, and z directions. Vector space uses column
        vectors.
        '''
        self.x_repeat = x_repeat
        self.y_repeat = y_repeat
        self.z_repeat = z_repeat
        self.unitcell = unit_cell
        self.vector_space = unit_cell.vector_space
        fractional = np.vstack([atom.fractional for atom in unit_cell.atoms])
        self.fractional = fractional
        self.cartesian = None
        self.repeat_atoms([1, 0, 0], x_repeat)
        self.repeat_atoms([0, 1, 0], y_repeat)
        self.repeat_atoms([0, 0, 1], z_repeat)
        self.a_side_vector = x_repeat*np.array(unit_cell.a_lattice_vector)
        self.b_side_vector = y_repeat*np.array(unit_cell.b_lattice_vector)
        self.c_side_vector = z_repeat*np.array(unit_cell.c_lattice_vector)
        self.a_side_length = norm(self.a_side_vector)
        self.b_side_length = norm(self.b_side_vector)
        self.c_side_length = norm(self.c_side_vector)

    def __repr__(self):
        return (f"SuperCell({repr(self.unitcell)}, {self.x_repeat}, "
                + f"{self.y_repeat}, {self.z_repeat})")

    def repeat_atoms(self, vector, repeat):
        '''
        Repeats the atoms within the current supercell using a vector to define
        the positions of the repeated atoms.
        '''
        new_atom_array = np.repeat(self.fractional, repeat)
        shift_array = np.array(vector)
        shift_array = np.array(
            [shift_array*index for index in range(0, repeat)])
        shift_array = shift_array[np.newaxis, :]
        shift_array = np.repeat(shift_array, self.fractional.shape[0], axis=0)
        shift_array = shift_array.reshape(
            shift_array.shape[0]*shift_array.shape[1], 3)
        new_atom_array['coordinates'] += shift_array
        new_atom_array['coordinates'] = np.around(
            new_atom_array['coordinates'], 6)
        self.fractional = new_atom_array

    def randomise(self, ratios):
        '''
        Randomises the atoms in the supercell structure. Randomisation is based
        on the provided ratio list. Ratios should add to one and be input in
        the order of the elements arranged alphabetically.
        Example: elements = (Fe, Ti, Pt), ratios = (0.3, 0.2, 0.5)
                 gives: Fe = 0.3, Pt = 0.2, Ti = 0.5.
        '''
        unique_elements = np.unique(self.fractional['element'])
        elements = np.random.choice(unique_elements, self.fractional.shape[0],
                                    p=ratios)
        self.fractional['element'] = elements

    def set_cartesian(self):
        '''
        Creates a cartesian coordinate set from the fractional set and
        corresponding vector space.
        '''
        coordinates = self.fractional['coordinates'].copy()
        coordinates = self.vector_space @ coordinates.T
        coordinates = coordinates.T
        coordinates = np.around(coordinates, 8)
        cartesian = self.fractional.copy()
        cartesian['coordinates'] = coordinates
        self.cartesian = cartesian

    def set_fractional(self):
        '''
        Creates a fractional coordinate set from the cartesian set using the
        supercell's inverse vector space to transform the coordinates.
        '''
        coordinates = self.cartesian['coordinates'].copy()
        coordinates = np.linalg.inv(self.vector_space) @ coordinates.T
        coordinates = coordinates.T
        coordinates = np.around(coordinates, 8)
        fractional = self.cartesian.copy()
        fractional['coordinates'] = coordinates
        self.fractional = fractional
