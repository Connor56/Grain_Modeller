import os
import sys
import unittest
import numpy as np
from copy import copy


class TestUnitcell(unittest.TestCase):

    def test_unitcell_instantiation(self):
        '''
        Test unitcell instantiates correctly.
        '''
        basis = [Atom('Pt', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.5)]
        lat_a, lat_b, lat_c = [2, 0, 0], [0, 2, 0], [0, 0, 2]
        test_unitcell = UnitCell(basis, lat_a, lat_b, lat_c)
        self.assertTrue(
            repr(test_unitcell)
            == "UnitCell([Atom('Pt', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.5"
            + ")], [2, 0, 0], [0, 2, 0], [0, 0, 2])")

    def test_unitcell_instantiation2(self):
        '''
        Check UnitCell vector space is created correctly. Vector spaces should
        be vector column spaces.
        '''
        basis = [Atom('Pt', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.5)]
        lat_a, lat_b, lat_c = [2, 0, 0], [0, 2, 0], [0, 0, 2]
        test_unitcell = UnitCell(basis, lat_a, lat_b, lat_c)
        vector_space = np.around(test_unitcell.vector_space, 6).tolist()
        self.assertTrue(vector_space == [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_check_vectors(self):
        '''
        Test the check vectors function correctly stops bad vectors from being
        used by the unitcell. Check it returns True or False depending on
        whether the vector is upper triangular.
        '''
        a, b, c = [1, 1.5, 0], [0.5, 2, 0], [0, 0, 2]
        self.assertWarnsRegex(
            UserWarning,
            "Lattice Vector a's primary component: x, is not its largest."
            + " Check this is intentional.",
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [4, 1.5, 0], [2.5, 2, 0], [0, 0, 2]
        self.assertWarnsRegex(
            UserWarning,
            "Lattice Vector b's primary component: y, is not its largest."
            + " Check this is intentional.",
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [4, 1.5, 0], [2.5, 5, 0], [0, 4, 2]
        self.assertWarnsRegex(
            UserWarning,
            "Lattice Vector c's primary component: z, is not its largest."
            + " Check this is intentional.",
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [4, 1.5, 0], [5, 6, 0], [0, 0, 2]
        self.assertRaisesRegex(
            ValueError,
            'Lattice Vector a, does not have the largest magnitude in its '
            + 'primary direction: x. Check your input or consider swapping '
            + 'vectors to remedy this.',
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [7, 6, 0], [3, 5, 0], [0, 0, 2]
        self.assertRaisesRegex(
            ValueError,
            'Lattice Vector b, does not have the largest magnitude in its '
            + 'primary direction: y. Check your input or consider swapping '
            + 'vectors to remedy this.',
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [4, 2, 0], [3, 6, 5], [0, 0, 2]
        self.assertRaisesRegex(
            ValueError,
            'Lattice Vector c, does not have the largest magnitude in its '
            + 'primary direction: z. Check your input or consider swapping '
            + 'vectors to remedy this.',
            UnitCell.check_vectors, '_', a, b, c)
        a, b, c = [1, 0.3, 0], [0.5, 2, 0], [0, 0, 2]
        self.assertTrue(UnitCell.check_vectors('_', a, b, c))
        a, b, c = [5, 0, 0], [0.5, 2, 0], [3, 1, 5]
        self.assertFalse(UnitCell.check_vectors('_', a, b, c))

    def test_alter_vectors(self):
        '''
        Does alter vectors correctly reformat the vector space into an upper
        triangular matrix? Does it bring orthoganal vectors back to alignment?
        '''
        basis = [Atom('Pt', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.5)]
        lat_a, lat_b, lat_c = [5, 1, 0], [0, 2, 0], [0, 0, 2]
        test_unitcell = UnitCell(basis, lat_a, lat_b, lat_c)
        vector_space = test_unitcell.vector_space
        expected_vector_space = np.array([[5.09902, 0.392232, 0.0],
                                          [0.0, 1.961161, 0.0],
                                          [0.0, 0.0, 2.0]])
        self.assertTrue(np.all(vector_space == expected_vector_space))
        expected_representation = (f"UnitCell({basis}, [5.09902, -0.0, 0.0], "
                                  + "[0.392232, 1.961161, 0.0], "
                                  + "[0.0, 0.0, 2.0])")
        self.assertTrue(repr(test_unitcell) == expected_representation)
        vector_space = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        matrix = linalg.rotation_matrix('z', np.pi/8)
        vector_space = matrix @ vector_space
        lat_a = vector_space.T[0].tolist()
        lat_b = vector_space.T[1].tolist()
        lat_c = vector_space.T[2].tolist()
        vector_space_before = vector_space
        test_unitcell = UnitCell(basis, lat_a, lat_b, lat_c)
        vector_space = test_unitcell.vector_space
        expected_vector_space = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.assertTrue(np.all(vector_space == expected_vector_space))
        self.assertFalse(np.all(vector_space_before == vector_space))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from unitcell import UnitCell
    import linear_algebra as linalg
    from atom import Atom
    unittest.main()
