import unittest
import os
import sys
import numpy as np
import cProfile
import pstats
from pstats import SortKey


class TestCrystallography(unittest.TestCase):

    def test_miller_to_intercepts(self):
        '''
        Does miller_to_intercepts correctly transform (hkl) plane indices into
        intercepts along the crystal axes?
        '''
        hkl_plane = [1, 1, 1]
        intercepts = crystallography.miller_to_intercepts(hkl_plane)
        expected_intercepts = np.array([1, 1, 1])
        self.assertTrue(np.all(intercepts == expected_intercepts))
        hkl_plane = [1, 2, 5]
        intercepts = crystallography.miller_to_intercepts(hkl_plane)
        expected_intercepts = np.array([1, 1/2, 1/5])
        self.assertTrue(np.all(intercepts == expected_intercepts))
        hkl_plane = [1, 1, 0]
        intercepts = crystallography.miller_to_intercepts(hkl_plane)
        expected_intercepts = np.array([1, 1, np.inf])
        self.assertTrue(np.all(intercepts == expected_intercepts))
        hkl_plane = [-5, 9, 1]
        intercepts = crystallography.miller_to_intercepts(hkl_plane)
        expected_intercepts = np.array([-1/5, 1/9, 1])
        self.assertTrue(np.all(intercepts == expected_intercepts))
        hkl_plane = [-5, 0, 1]
        intercepts = crystallography.miller_to_intercepts(hkl_plane)
        expected_intercepts = np.array([-1/5, np.inf, 1])
        self.assertTrue(np.all(intercepts == expected_intercepts))

    def test_cartesian_plane_normal(self):
        '''
        Is the correct plane normal returned when inputting a vector space and
        a plane in hkl format?
        '''
        # Test for a simple hkl plane and orthogonal vector space
        vector_space = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]]).T
        hkl_plane = [1, 0, 0]
        plane_normal = crystallography.cartesian_plane_normal(
            vector_space, hkl_plane)
        expected_plane_normal = np.array([1, 0, 0])
        self.assertTrue(
            np.all(np.isclose(plane_normal, expected_plane_normal, atol=1e-8)))

        # Test for a complex hkl plane and orthogonal vector space
        hkl_plane = [1, -1, 1]
        plane_normal = crystallography.cartesian_plane_normal(
            vector_space, hkl_plane)
        expected_plane_normal = np.array([1, -0.5, 1])/np.sqrt(2.25)
        self.assertTrue(
            np.all(np.isclose(plane_normal, expected_plane_normal, atol=1e-8)))

        # Test for a simple hkl plane and complex vector space
        hkl_plane = [1, 0, 0]
        vector_space = np.array([[1, 0, 0], [0.5, 2, 0], [0.5, 1, 1]]).T
        plane_normal = crystallography.cartesian_plane_normal(
            vector_space, hkl_plane)
        expected_plane_normal = np.array([2, -0.5, -0.5])/np.sqrt(4.5)
        self.assertTrue(
            np.all(np.isclose(plane_normal, expected_plane_normal, atol=1e-8)))

    def test_distance_symmetries_no_box(self):
        '''
        Test that the correct distance symmetries are calculated from the atoms
        inside the supercell cube with no box selection.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(
            test_basis, [3.82, 0, 0], [0, 3.82, 0], [0, 0, 3.711])
        supercell = SuperCell(unitcell, 20, 20, 20)
        supercell.set_cartesian()
        distance_symmetries = crystallography.distance_symmetries(
            supercell.cartesian['coordinates'])
        expected_distance_symmetries = np.array(
            [[0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889,
             2.662889, 2.662889, 2.701148, 2.701148, 2.701148, 2.701148],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889,
              2.701148, 2.701148, 3.711, 3.711, 3.82, 3.82],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889,
              2.701148, 2.701148, 3.711, 3.82, 3.82, 3.82],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889,
              2.701148, 2.701148, 3.711, 3.82, 3.82, 4.589959],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.701148, 2.701148,
              2.701148, 2.701148, 3.711, 3.82, 3.82, 3.82],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.701148, 2.701148,
              2.701148, 2.701148, 3.711, 3.82, 3.82, 4.589959],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.701148, 3.711,
              3.711, 3.82, 3.82, 4.589959, 4.589959, 4.656542],
             [0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.701148, 3.711,
              3.82, 3.82, 4.589959, 4.656542, 4.656542, 4.656542],
             [0.0, 2.662889, 2.662889, 2.662889, 2.701148, 2.701148, 3.711,
              3.82, 3.82, 3.82, 4.589959, 4.589959, 4.656542],
             [0.0, 2.662889, 2.662889, 2.662889, 2.701148, 2.701148, 3.711,
              3.82, 3.82, 4.589959, 4.589959, 4.656542, 4.656542],
             [0.0, 2.662889, 2.662889, 2.701148, 3.711, 3.82, 3.82, 4.589959,
              4.656542, 4.656542, 5.325779, 5.325779, 5.402296]])
        self.assertTrue(
            np.all(expected_distance_symmetries == distance_symmetries))

    def test_distance_symmetries_with_box(self):
        '''
        Test that the correct distance symmetries are calculated from the atoms
        inside the supercell cube with box selection. Check for different
        neighbour choices also.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(
            test_basis, [3.82, 0, 0], [0, 3.82, 0], [0, 0, 3.711])
        supercell = SuperCell(unitcell, 20, 20, 20)
        supercell.set_cartesian()
        box = [[10, 50], [10, 50], [10, 50]]
        distance_symmetries = crystallography.distance_symmetries(
            supercell.cartesian['coordinates'], box=box)
        expected_distance_symmetries = np.array(
            [[0.0, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889, 2.662889,
             2.662889, 2.662889, 2.701148, 2.701148, 2.701148, 2.701148]])
        self.assertTrue(np.all(
            expected_distance_symmetries == distance_symmetries))
        distance_symmetries = crystallography.distance_symmetries(
            supercell.cartesian['coordinates'], box=box, neighbours=8)
        self.assertTrue(np.all(
            expected_distance_symmetries[:, :-4] == distance_symmetries))

    def test_distance_symmetries_very_simple(self):
        '''
        Check that for a very simple set of points this algorithm still works
        correctly.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 40, 40, 40)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[10, 20], [10, 20], [10, 20]]
        symmetries = crystallography.distance_symmetries(atoms, box=box)
        expected_symmetries = np.array(
            [[0, 1, 1, 1, 1, 1, 1, 1.414214, 1.414214, 1.414214, 1.414214,
              1.414214, 1.414214]])
        self.assertTrue(np.all(symmetries == expected_symmetries))

    def test_distance_symmetries_very_simple(self):
        '''
        Check that for a very simple set of points this algorithm still works
        correctly.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 2, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 40, 40, 40)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[10, 30], [10, 30], [10, 30]]
        symmetries = crystallography.distance_symmetries(atoms, box=box)
        expected_symmetries = np.array(
            [[0, 1, 1, 1, 1, 1.414214, 1.414214, 1.414214, 1.414214, 2, 2,
              2, 2]])
        self.assertTrue(np.all(symmetries == expected_symmetries))

    def test_neighbour_distances(self):
        '''
        Test neighbour distances for a simple atom array.
        '''
        atoms = np.array([[1, 1, 1], [-1, 2, 3], [2, 1, 2]])
        distances = crystallography.neighbour_distances(atoms, 1)
        expected_distances = np.array([[0, 1.414214], [0, 3], [0, 1.414214]])
        self.assertTrue(np.all(distances == expected_distances))







if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import testing_tools
    import crystallography
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    unittest.main()
