import os
import sys
import unittest
import numpy as np
from copy import copy


class TestTransfroms(unittest.TestCase):

    def test_rotate(self):
        '''
        Does rotate correctly rotate an atom or supercell structure?
        '''
        angle = np.pi/2
        matrix = linalg.rotation_matrix('y', -angle)
        atom = Atom('Fe', 1, 0, 0)
        representation_before = repr(atom)
        transforms.rotate(atom, matrix)
        representation_after = repr(atom)
        self.assertFalse(representation_before == representation_after)
        self.assertTrue(representation_before == "Atom('Fe', 1.0, 0.0, 0.0)")
        self.assertTrue(representation_after == "Atom('Fe', 0.0, 0.0, 1.0)")

    def test_rotate_atom(self):
        '''
        Test an atom object is correctly rotated.
        '''
        angle = np.pi/2
        matrix = linalg.rotation_matrix('y', -angle)
        atom = Atom('Fe', 1, 0, 0)
        representation_before = repr(atom)
        transforms.rotate_atom(atom, matrix)
        representation_after = repr(atom)
        self.assertFalse(representation_before == representation_after)
        self.assertTrue(representation_before == "Atom('Fe', 1.0, 0.0, 0.0)")
        self.assertTrue(representation_after == "Atom('Fe', 0.0, 0.0, 1.0)")

    def test_rotate_supercell(self):
        '''
        Test that rotate supercell correctly changes the vector space and
        lattice parameters of a given supercell.
        '''
        angle = np.pi
        matrix = linalg.rotation_matrix('y', -angle)
        basis = [Atom('Fe', 1, 0, 0)]
        unitcell = UnitCell(basis, [2, 0, 0], [0, 2, 0], [0, 0, 2])
        supercell = SuperCell(unitcell, 2, 1, 2)
        transforms.rotate_supercell(supercell, matrix)
        expected_vector_space = [[-2, 0, 0], [0, 2, 0], [0, 0, -2]]
        vector_space = np.around(supercell.vector_space, 6)
        self.assertTrue(vector_space.tolist() == expected_vector_space)
        self.assertTrue(
            np.around(supercell.a_side_vector, 6).tolist() == [-4, 0, 0])
        self.assertTrue(
            np.around(supercell.b_side_vector, 6).tolist() == [0, 2, 0])
        self.assertTrue(
            np.around(supercell.c_side_vector, 6).tolist() == [0, 0, -4])

    def test_rotate_supercell_correctly_alters_cartesian(self):
        '''
        Test that rotate supercell correctly changes cartesian coordinates
        when rotating a supercell that already has them.
        '''
        angle = np.pi
        matrix = linalg.rotation_matrix('y', -angle)
        basis = [Atom('Fe', 1, 0, 0)]
        unitcell = UnitCell(basis, [2, 0, 0], [0, 2, 0], [0, 0, 2])
        supercell = SuperCell(unitcell, 2, 1, 2)
        supercell.set_cartesian()
        cartesian_coordinates = supercell.cartesian['coordinates'].tolist()
        expected_cartesian_coordinates = [[2, 0, 0], [2, 0, 2],
                                          [4, 0, 0], [4, 0, 2]]
        self.assertTrue(
            cartesian_coordinates == expected_cartesian_coordinates)
        transforms.rotate_supercell(supercell, matrix)
        expected_vector_space = [[-2, 0, 0], [0, 2, 0], [0, 0, -2]]
        vector_space = np.around(supercell.vector_space, 6).tolist()
        self.assertTrue(vector_space == expected_vector_space)
        cartesian_coordinates = np.around(
            supercell.cartesian['coordinates'], 6).tolist()
        expected_cartesian_coordinates = [[-2, 0, 0], [-2, 0, -2],
                                          [-4, 0, 0], [-4, 0, -2]]
        self.assertTrue(
            cartesian_coordinates == expected_cartesian_coordinates)

    def OFF_test_rotate_supercell_visual_inspection(self):
        '''
        Rotates a test supercell and visualises it to see if it work correctly.
        '''
        basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0, 0.5, 0.5),
                 Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(basis, [3.83, 0, 0], [0, 3.83, 0], [0, 0, 3.711])
        supercell = SuperCell(unitcell, 10, 10, 10)
        angle = np.pi/4
        matrix = linalg.rotation_matrix('y', -angle)
        transforms.rotate_supercell(supercell, matrix)
        supercell.set_cartesian()
        cartesian_coordinates = supercell.cartesian["coordinates"].tolist()
        testing_tools.xyz_file_output('rotation_test', cartesian_coordinates)

    def test_translate(self):
        '''
        Does translate correctly move an atom and a supercell? Including
        updating cartesian coordinates if any exist, or using cartesian as the
        basis of translation.
        '''
        test_atoms = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_atoms, [2, 0, 0], [0, 2, 0], [0, 0, 2])
        supercell = SuperCell(unitcell, 1, 1, 1)
        transforms.translate(supercell, [1, 1, 1])
        array = supercell.fractional
        expected_array = np.array(
            [('Fe', [1., 1, 1]), ('Pt', [1.5, 1.5, 1.5])],
            dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])
        # True for simple fractional move
        self.assertTrue(np.all(array == expected_array))
        transforms.translate(supercell, [-1, -1, -1])
        transforms.translate(supercell, [2, 2, 2], coordinates='cartesian')
        array = supercell.fractional
        # True for simple cartesian move
        self.assertTrue(np.all(array == expected_array))
        transforms.translate(supercell, [1.5, 3, 1], coordinates='cartesian')
        expected_array = np.array(
            [('Fe', [1.75, 2.5, 1.5]), ('Pt', [2.25, 3.0, 2.0])],
            dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])
        # True for complex cartesian move
        self.assertTrue(np.all(array == expected_array))
        test_atoms = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_atoms, [2, 0, 0], [1, 2, 0], [1, 1, 2])
        supercell = SuperCell(unitcell, 1, 1, 1)
        supercell.set_cartesian()
        transforms.translate(supercell, [1, 1, 1])
        array = supercell.cartesian
        expected_array = np.array(
            [('Fe', [4, 3, 2]), ('Pt', [6, 4.5, 3])],
            dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])
        # True for complex move with complex vector space and cartesian update
        self.assertTrue(np.all(array == expected_array))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import transforms
    import linear_algebra as linalg
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    import testing_tools
    unittest.main()
