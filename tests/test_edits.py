import unittest
import os
import sys
import numpy as np

#from grain_modeller.linear_algebra import rotation_matrix


class TestEdits(unittest.TestCase):

    def test_cut_initialisation(self):
        '''
        Does the Cut class initialise correctly?
        '''
        cut_1 = edits.Cut('s', [1, 1, 1], radius=4)
        cut_2 = edits.Cut('p', [1, 1, 1], plane=[1, 1, 1])
        expected_1 = ("Cut('s', [1, 1, 1], plane=None, radius=4, out=True, "
                      "axes=None)")
        expected_2 = ("Cut('p', [1, 1, 1], plane=[1, 1, 1], radius=None, "
                      "out=True, axes=None)")
        self.assertTrue(repr(cut_1) == expected_1)
        self.assertTrue(repr(cut_2) == expected_2)
        self.assertRaisesRegex(
            ValueError, "Unknown cut type: 'LAX'.",
            edits.Cut, 'LAX', [1, 1, 1], plane=[1, 1, 1])
        self.assertRaisesRegex(
            ValueError, "Point: 'not a point', is not a list",
            edits.Cut, 'p', 'not a point', plane=[1, 1, 1])
        self.assertRaises(ValueError, edits.Cut, 'p', [1, 1], plane=[1, 1, 1])

    def test_twin_initialisation(self):
        '''
        Does the twin data structure initialise correctly? Particularly, if
        two vectors are added that are not orthogonal.
        '''
        plane_point = np.array([1, 1, 1])
        normal_vector = np.array([1, 1, 1])
        rotation_vector = np.array([0, 0, 1])
        angle = np.deg2rad(50)
        self.assertRaisesRegex(ValueError,
            f"Normal vector and rotation vector must be orthogonal. Vectors "
            f"\[1 1 1\] and \[0 0 1\] are not orthogonal.",
            edits.Twin, plane_point, normal_vector, rotation_vector, angle)

    def test_plane_cut(self):
        '''
        Check plane_cut function correctly deletes atoms from supercell.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        atoms_before = supercell.fractional
        cut = edits.Cut('p', [2, 5, 1], plane=[1, 0, 0])
        edits.plane_cut(supercell, cut)
        atoms = supercell.fractional
        self.assertFalse(atoms.shape[0] == atoms_before.shape[0])
        atoms = atoms['coordinates']
        self.assertTrue(np.all(atoms[:, 0] <= 2))
        self.assertFalse(np.all(atoms[:, 1] <= 5))
        self.assertFalse(np.all(atoms[:, 2] <= 1))

    def test_cartesian_plane_cut(self):
        '''
        Test a cartesian plane cut appropriately deletes atoms from the
        fractional supercell and updates the cartesian coordinates
        appropriately.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        atoms_before = supercell.fractional
        cut = edits.Cut('cp', [3, 2, 25], plane=[0, 0, 1])
        edits.cartesian_plane_cut(supercell, cut)
        atoms = supercell.fractional
        self.assertFalse(atoms.shape[0] == atoms_before.shape[0])
        supercell.set_cartesian()
        atoms = supercell.cartesian
        atoms = atoms['coordinates']
        self.assertFalse(np.all(atoms[:, 0] <= 3))
        self.assertFalse(np.all(atoms[:, 1] <= 2))
        self.assertTrue(np.all(atoms[:, 2] <= 25))

    def test_cartesian_plane_cut_complicated_unitcell_and_plane(self):
        '''
        Test a cartesian plane cut appropriately deletes atoms from the
        fractional supercell and updates.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        atoms_before = supercell.fractional
        cut = edits.Cut('cp', [3, 2, 25], plane=[0, 0, 1])
        edits.cartesian_plane_cut(supercell, cut)
        atoms = supercell.fractional
        self.assertFalse(atoms.shape[0] == atoms_before.shape[0])
        supercell.set_cartesian()
        atoms = supercell.cartesian
        atoms = atoms['coordinates']
        self.assertFalse(np.all(atoms[:, 0] <= 3))
        self.assertFalse(np.all(atoms[:, 1] <= 2))
        self.assertTrue(np.all(atoms[:, 2] <= 25))
    
    def test_cartesian_plane_cut_with_rotation(self):
        '''
        Test a cartesian plane cut still works as expected when used on a
        supercell which has been rotated.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        vector = np.array([0, 1, 0])
        atoms_before = supercell.fractional
        rotation_matrix = linalg.vector_rotation_matrix(np.pi/4, vector)
        transforms.rotate(supercell, rotation_matrix)
        cut = edits.Cut('cp', [0, 0, 0], plane=[1, 0, 0])
        edits.cartesian_plane_cut(supercell, cut)
        atoms = supercell.fractional
        self.assertFalse(atoms.shape[0] == atoms_before.shape[0])
        supercell.set_cartesian()
        atoms = supercell.cartesian
        atoms = atoms['coordinates']
        self.assertTrue(np.all(atoms[:, 2] <= 0))

    def test_spherical_cut(self):
        '''
        Does spherical cut function correctlly delete atoms from supercell?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        atoms_before = supercell.fractional
        point = [5, 5, 5]
        cut = edits.Cut('s', point, radius=3)
        point = np.array(point)
        edits.spherical_cut(supercell, cut)
        atoms = supercell.fractional
        self.assertFalse(atoms.shape[0] == atoms_before.shape[0])
        atoms = atoms['coordinates']
        atoms = np.linalg.norm(atoms - point, axis=1)
        self.assertTrue(np.all(atoms >= 3))
        supercell = SuperCell(unitcell, 10, 10, 10)
        cut = edits.Cut('s', point, radius=3, out=False)
        point = np.array(point)
        edits.spherical_cut(supercell, cut)
        atoms = supercell.fractional
        atoms = atoms['coordinates']
        atoms = np.linalg.norm(atoms - point, axis=1)
        self.assertTrue(np.all(atoms <= 3))

    def test_ellipsoid_cut(self):
        '''
        Does ellipsoid cut correctly delete atoms?
        '''

    def test_reflect(self):
        '''
        Does a simple supercell get reflected correctly across a simple plane?
        '''
        test_basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(test_basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 1, 1, 1)
        reflection = edits.Reflection(
            np.array([1, 0, 0]), np.array([1, 0, 0]), out=True)
        edits.reflect(supercell, reflection)
        coordinates = supercell.fractional['coordinates']
        expected_coordinates = np.array([[0, 0, 0], [2, 0, 0]])
        self.assertTrue(np.all(coordinates == expected_coordinates))

    def test_reflect_more_complicated_supercell(self):
        '''
        Does a slightly more complicated supercell get reflected correctly
        across a simple plane?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 1, 1, 1)
        reflection = edits.Reflection(
            np.array([1, 0, 0]), np.array([1, 0, 0]), out=True)
        edits.reflect(supercell, reflection)
        coordinates = supercell.fractional['coordinates']
        expected_coordinates = np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [2, 0, 0], [1.5, 0.5, 0.5]])
        self.assertTrue(np.all(coordinates == expected_coordinates))

    def test_reflect_gets_rid_of_duplicate_atoms(self):
        '''
        Does a slightly more complicated supercell get reflected correctly
        across a more complicated plane?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 1, 1, 1)
        reflection = edits.Reflection(
            np.array([0.5, 0, 0]), np.array([1, 0, 0]), out=True)
        edits.reflect(supercell, reflection)
        coordinates = supercell.fractional['coordinates']
        expected_coordinates = np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0]])
        self.assertTrue(np.all(coordinates == expected_coordinates))

    def test_reflect_more_complicated_reflection_vector(self):
        '''
        Does a slightly more complicated supercell get reflected correctly
        across a more complicated plane?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 1, 1, 1)
        reflection = edits.Reflection(
            np.array([1, 0, 0]), np.array([1, 1, 0]), out=True)
        edits.reflect(supercell, reflection)
        coordinates = supercell.fractional['coordinates']
        expected_coordinates = np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 0]])
        self.assertTrue(np.all(coordinates == expected_coordinates))

    def test_reflect_more_complicated_reflection_vector_and_supercell(self):
        '''
        Does a more complicated trigonal supercell get reflected correctly
        across a simple plane when using cartesian coordinates also?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [1, 0, 0], [0.5, 3, 0], [0, 0.1, 1])
        supercell = SuperCell(unitcell, 1, 1, 1)
        supercell.set_cartesian()
        reflection = edits.Reflection(
            np.array([1, 0, 0]), np.array([1, 0, 0]), out=True)
        edits.reflect(supercell, reflection)
        coordinates = supercell.fractional['coordinates']
        expected_coordinates = np.array(
            [[0, 0, 0], [0, 0.5, 0.5], [2, -0.10915969, 0.03242367],
             [2, 0.39084031, 0.53242367]])
        self.assertTrue(np.all(coordinates == expected_coordinates))
        cartesian_coordinates = supercell.cartesian['coordinates']
        expected_coordinates = np.array(
            [[0, 0, 0], [0.25, 1.55, 0.5],
             [1.94542016, -0.32423669, 0.03242367],
             [2.19542016, 1.22576331, 0.53242367]])
        self.assertTrue(np.all(np.isclose(
            cartesian_coordinates, expected_coordinates, atol=1e-8)))

    def OFF_test_reflect_very_complicated_rt12_supercell(self):
        '''
        Tests that the complicated ndfe12 unitcell with a complicated (011)
        reflection plane creates the correct
        '''
        test_basis = [
            Atom('Nd', 0.000000, 0.000000, 0.000000),
            Atom('Nd', 0.500000, 0.500000, 0.500000),
            Atom('Fe', 0.356800, 0.000000, 0.000000),
            Atom('Fe', 0.856800, 0.500000, 0.5000),
            Atom('Fe', 0.643200, 0.000000, 0.00000),
            Atom('Fe', 0.143200, 0.500000, 0.5000),
            Atom('Fe', 0.000000, 0.643200, 0.00000),
            Atom('Fe', 0.500000, 0.143200, 0.5000),
            Atom('Fe', 0.000000, 0.356800, 0.00000),
            Atom('Fe', 0.500000, 0.856800, 0.5000),
            Atom('Fe', 0.272800, 0.500000, 0.00000),
            Atom('Fe', 0.772800, 0.000000, 0.50000),
            Atom('Fe', 0.227200, 0.000000, 0.50000),
            Atom('Fe', 0.727200, 0.500000, 0.00000),
            Atom('Fe', 0.000000, 0.227200, 0.50000),
            Atom('Fe', 0.500000, 0.727200, 0.00000),
            Atom('Fe', 0.500000, 0.272800, 0.000000),
            Atom('Fe', 0.000000, 0.772800, 0.50000),
            Atom('Fe', 0.250000, 0.250000, 0.2500),
            Atom('Fe', 0.750000, 0.750000, 0.7500),
            Atom('Fe', 0.250000, 0.250000, 0.7500),
            Atom('Fe', 0.750000, 0.750000, 0.2500),
            Atom('Fe', 0.750000, 0.250000, 0.2500),
            Atom('Fe', 0.250000, 0.750000, 0.7500),
            Atom('Fe', 0.250000, 0.750000, 0.2500),
            Atom('Fe', 0.750000, 0.250000, 0.7500)]
        unitcell = UnitCell(
            test_basis, [8.611, 0, 0], [0, 8.611, 0], [0, 0, 4.802])
        supercell = SuperCell(unitcell, 10, 10, 10)
        reflection = edits.Reflection(
            np.array([0, 0, 10]), np.array([0, 1, 1]), out=True)
        edits.reflect(supercell, reflection)
        supercell.set_fractional()
        supercell.set_cartesian()
        test_tool.xyz_output('rt12_reflection', supercell.cartesian)
        self.assertTrue(False)

    def OFF_test_create_twin(self):
        '''
        Test that a twin can be created using a simple crystal, a simple
        rotation vector, and a simple angle through which to rotate. Do this
        using a BCC structure.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [2.8, 0, 0], [0, 2.8, 0], [0, 0, 2.8])
        supercell = SuperCell(unitcell, 2, 2, 2)
        plane_point = np.array([1, 1, 1])
        normal_vector = np.array([1, 1, 1])
        rotation_vector = np.array([-1, 0, 1])
        angle = np.deg2rad(50)
        twin = edits.Twin(plane_point, normal_vector, rotation_vector, angle)
        edits.create_twin(supercell, twin)






if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    import edits
    import linear_algebra as linalg
    import transforms
    import testing_tools as test_tool
    import crystallography
    unittest.main()
