'''
Name:
    Atom
Description:
    Contains the atom class, which defines a custom data object representing a
    single atom.
'''
import numpy as np


class Atom():

    def __init__(self, element, x_coord, y_coord, z_coord):
        '''
        Instantiate new atom object.
        '''
        #Check initialisation parameters are reasonable
        if not isinstance(element, str):
            raise ValueError('Element must be a string.')
        if not isinstance(x_coord, int) and not isinstance(x_coord, float):
            raise ValueError('All coordinates must be numeric, x is not.')
        if not isinstance(y_coord, int) and not isinstance(y_coord, float):
            raise ValueError('All coordinates must be numeric, y is not.')
        if not isinstance(z_coord, int) and not isinstance(z_coord, float):
            raise ValueError('All coordinates must be numeric, z is not.')
        self.fractional = np.array(
            [(element, [x_coord, y_coord, z_coord])],
            dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])
        self.cartesian = None
        self.element = element

    def __repr__(self):
        x, y, z = self.fractional['coordinates'][0].tolist()
        return f"Atom('{self.element}', {x}, {y}, {z})"

    def __copy__(self):
        x, y, z = self.fractional['coordinates'][0].tolist()
        return Atom(self.element, x, y, z)

    def set_cartesian(self, vector_space):
        '''
        Set the cartesian coordinates of the atom, based on a vector space -
        generally that of a unitcell or supercell. Requires column vectors for
        the vector space.
        '''
        fractional = self.fractional['coordinates'][0]
        cartesian = vector_space @ fractional.T
        self.cartesian = np.array(
            [(self.element, cartesian)],
            dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])
