'''
Name:
    Unitcell
Description:
    Contains the unitcell class, which defines a custom data object
    representing a single unitcell.
'''

from warnings import warn
import numpy as np
from numpy.linalg import norm
import linear_algebra as linalg
from atom import Atom


class UnitCell():

    def __init__(self, atoms, a_lat_vector, b_lat_vector, c_lat_vector):
        '''
        Instantiate a new UnitCell object, with a basis_list of Atoms, and
        three lattice vectors. Vector space uses column vectors.
        '''
        if not all([isinstance(atom, Atom) for atom in atoms]):
            raise TypeError(
                'Cannot create UnitCell: Basis must be a list of the class '
                + 'Atom.')
        alter = self.check_vectors(a_lat_vector, b_lat_vector, c_lat_vector)
        self.a_lattice_vector = a_lat_vector
        self.b_lattice_vector = b_lat_vector
        self.c_lattice_vector = c_lat_vector
        vector_space = np.vstack((a_lat_vector, b_lat_vector, c_lat_vector)).T
        self.vector_space = vector_space
        if alter:
            self.alter_vectors()
        self.a_lattice_parameter = norm(self.a_lattice_vector)
        self.b_lattice_parameter = norm(self.b_lattice_vector)
        self.c_lattice_parameter = norm(self.c_lattice_vector)
        self.atoms = atoms

    def check_vectors(self, lat_a, lat_b, lat_c):
        '''
        Check given unitcell lattice vectors are largest in their primary
        direction out of all the vectors. Warn if vector's primary component
        isn't its largest.
        '''
        lat_a = np.array(lat_a)
        lat_b = np.array(lat_b)
        lat_c = np.array(lat_c)
        all_lat = np.vstack((lat_a, lat_b, lat_c))
        if np.argsort(lat_a)[-1] != 0:
            warn("Lattice Vector a's primary component: x, is not its largest."
                 + " Check this is intentional.")
        if np.argsort(lat_b)[-1] != 1:
            warn("Lattice Vector b's primary component: y, is not its largest."
                 + " Check this is intentional.")
        if np.argsort(lat_c)[-1] != 2:
            warn("Lattice Vector c's primary component: z, is not its largest."
                 + " Check this is intentional.")
        if np.argsort(all_lat[:, 0])[-1] != 0:
            raise ValueError(
                'Lattice Vector a, does not have the largest magnitude in its '
                + 'primary direction: x. Check your input or consider '
                + 'swapping vectors to remedy this.')
        if np.argsort(all_lat[:, 1])[-1] != 1:
            raise ValueError(
                'Lattice Vector b, does not have the largest magnitude in its '
                + 'primary direction: y. Check your input or consider '
                + 'swapping vectors to remedy this.')
        if np.argsort(all_lat[:, 2])[-1] != 2:
            raise ValueError(
                'Lattice Vector c, does not have the largest magnitude in its '
                + 'primary direction: z. Check your input or consider '
                + 'swapping vectors to remedy this.')
        if np.all(np.triu(all_lat.T) == all_lat.T):
            return False
        else:
            return True

    def alter_vectors(self):
        '''
        Make vector space upper triangular through rotations. Use the
        transformation matrix this requires to alter the fractional coordinates
        of the atoms.
        '''
        vector_space = self.vector_space
        xy_rot = -np.arctan(vector_space[1, 0]/vector_space[0, 0])
        xy_rot = linalg.rotation_matrix('z', xy_rot)
        vector_space = xy_rot @ vector_space
        xz_rot = np.arctan(vector_space[2, 0]/vector_space[0, 0])
        xz_rot = linalg.rotation_matrix('y', xz_rot)
        vector_space = xz_rot @ vector_space
        yz_rot = -np.arctan(vector_space[2, 1]/vector_space[1, 1])
        yz_rot = linalg.rotation_matrix('x', yz_rot)
        vector_space = yz_rot @ vector_space
        rotation_matrix = yz_rot @ xz_rot @ xy_rot
        self.vector_space = np.around(vector_space, 6)
        self.a_lattice_vector = self.vector_space.T[0].tolist()
        self.b_lattice_vector = self.vector_space.T[1].tolist()
        self.c_lattice_vector = self.vector_space.T[2].tolist()
        return rotation_matrix

    def __repr__(self):
        '''
        UnitCell representation.
        '''
        basis = str(self.atoms)
        lat_a = str(self.a_lattice_vector)
        lat_b = str(self.b_lattice_vector)
        lat_c = str(self.c_lattice_vector)
        return "UnitCell(%s, %s, %s, %s)" % (basis, lat_a, lat_b, lat_c)
