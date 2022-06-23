'''
Name:
    Test Surface Creation
Description:
    Unittests for the Surface Creation module.
'''

import unittest
import os
import sys
import numpy as np
import cProfile
import pstats
from pstats import SortKey


class TestSurfaceCreation(unittest.TestCase):

    def test_reorientate_supercell_simple_supercell(self):
        '''
        Tests the reorientate_supercell correctly places the desired plane
        into the [001] direction.
        '''
        basis_list = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(
            basis_list, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 10, 10, 10)
        vector_space_before = supercell.vector_space.tolist()
        plane = (1, 1, 1)
        surface_creation.reorientate_supercell(supercell, plane)
        self.assertTrue(supercell.vector_space.tolist() != vector_space_before)
        vector_space = np.around(
            supercell.vector_space @ np.array([1, 1, 1]), 6)
        self.assertTrue(np.all(vector_space == np.array([0, 0, 1.732051])))

    def test_reorientate_supercell(self):
        '''
        Tests the reorientate_supercell correctly places the desired plane
        into the [001] direction for a more complicated unitcell.
        '''
        basis_list = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(
            basis_list, [3.83, 0, 0], [0, 3.83, 0], [0, 0, 3.711])
        supercell = SuperCell(unitcell, 10, 10, 10)
        vector_space_before = supercell.vector_space.tolist()
        fractional_before = supercell.fractional
        plane = (1, 1, 1)
        surface_creation.reorientate_supercell(supercell, plane)
        self.assertTrue(supercell.vector_space.tolist() != vector_space_before)
        plane_after = linalg.plane_normal(supercell.vector_space, plane)
        self.assertTrue(np.isclose(plane_after, np.array([0, 0, 1])).all())
        self.assertTrue((fractional_before == supercell.fractional).all())
        supercell.set_cartesian()
        cut = edits.Cut('cp', [0, 0, 14], [0, 0, 1])
        edits.make_cut(supercell, cut)
        atoms = supercell.cartesian['coordinates']
        self.assertTrue(np.sum(atoms[:, 2] >= 13.1) == 91)

    def test_reorientate_supercell_simple_change(self):
        '''
        Tests the reorientate_supercell correctly places the desired plane
        into the [001] direction for a more complicated unitcell.
        '''
        basis_list = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(
            basis_list, [4, 0, 0], [0, 4, 0], [0, 0, 4])
        supercell = SuperCell(unitcell, 10, 10, 10)
        vector_space_before = supercell.vector_space.tolist()
        fractional_before = supercell.fractional
        plane = (1, 0, 0)
        surface_creation.reorientate_supercell(supercell, plane)
        self.assertTrue(supercell.vector_space.tolist() != vector_space_before)
        intercepts = crystallography.miller_to_intercepts(plane)
        plane_after = linalg.plane_normal(supercell.vector_space, intercepts)
        self.assertTrue(np.isclose(plane_after, np.array([0, 0, 1])).all())
        self.assertTrue((fractional_before == supercell.fractional).all())
        supercell.set_cartesian()
        cut = edits.Cut('cp', [0, 0, 14], [0, 0, 1])
        edits.make_cut(supercell, cut)
        atoms = supercell.cartesian['coordinates']
        self.assertTrue(np.sum(atoms[:, 2] >= 13.9) == 100)

    def test_get_rotation_vector(self):
        '''
        Does get rotation vector return the expected plane normal around which
        to perform the optimal rotation.
        '''
        direction = np.array([1, 0, 0])
        expected_normal = [0, -1, 0]
        z = np.array([0, 0, 1])
        normal = surface_creation.get_rotation_vector(direction, z)
        self.assertTrue(normal.tolist() == expected_normal)
        direction = np.array([1, 1, 1])
        direction = direction/np.linalg.norm(direction)
        normal = surface_creation.get_rotation_vector(direction, z)
        expected_normal = np.array([1/3, -1/3, 0])
        expected_normal = expected_normal/np.linalg.norm(expected_normal)
        normal = np.around(normal, 12)
        expected_normal = np.around(expected_normal, 12)
        self.assertTrue(normal.tolist() == expected_normal.tolist())

    def OFF_test_cut_slab(self):
        '''
        Tests a simple supercell can be turned into a cuboid shaped slab of the
        desired thickness.
        '''
        basis_list = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis_list, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        supercell = surface_creation.cut_slab(supercell, 10)
        atoms = supercell.fractional['coordinates']
        self.assertTrue(np.all(atoms[:, 2] >= 5))
        atoms = supercell.cartesian['coordinates']
        self.assertTrue(np.all(atoms[:, 2] >= 15))

    def OFF_test_cut_slab_complicated_111(self):
        '''
        Tests a complicated supercell can be turned into a cuboid shaped slab
        of the desired thickness. Even if it has been reorientated into the
        [111] direction.
        '''
        basis_list = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(basis_list, [3.83, 0, 0], [0, 3.83, 0],
                            [0, 0, 3.711])
        supercell = SuperCell(unitcell, 30, 30, 30)
        surface_creation.reorientate_supercell(supercell, [1, 1, 1])
        supercell = surface_creation.cut_slab(supercell, 10)
        atoms = supercell.cartesian['coordinates']
        self.assertTrue(np.around(np.min(atoms[:, 0]), 4) == -51.7528)
        self.assertTrue(np.around(np.max(atoms[:, 0]), 4) == 56.1508)
        self.assertTrue(np.around(np.min(atoms[:, 1]), 4) == -51.7685)
        self.assertTrue(np.around(np.max(atoms[:, 1]), 4) == 56.1524)

    def test_select_working_volume_selects_the_correct_slab(self):
        '''
        Is the correct slab selected to be the volume which is fit to the
        desired thickness.
        '''
        basis_list = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis_list, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        supercell.set_cartesian()
        points = supercell.cartesian['coordinates']
        slab = surface_creation.select_working_volume(points, 10)
        self.assertTrue(slab.number_of_points == 400)
        self.assertTrue(slab.volume == 7290.0)
        self.assertTrue(slab.z_min == 15)
        self.assertTrue(slab.z_max == 25)

    def test_split_into_slabs(self):
        '''
        Is a volume correctly split into slabs of the desired thickness?
        '''
        basis_list = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis_list, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        supercell.set_cartesian()
        points = supercell.cartesian['coordinates']
        z_points = np.arange(0, 10, 3)
        slabs = surface_creation.split_into_slabs(
            points, 10, z_points=z_points)
        self.assertTrue(len(slabs) == 4)
        self.assertTrue(
            np.all([x.number_of_points == 400 for x in slabs]))
        self.assertTrue(
            np.all([x.point_density == slabs[0].point_density for x in slabs]))

    def test_get_point_density(self):
        '''
        Does it find the correct point density given some points and a volume.
        '''
        point_density = surface_creation.get_point_density(100, 10)
        self.assertTrue(point_density == 10)

    def OFF_test_get_xy_dimensions_finds_correct_xy_values_simple(self):
        '''
        Given some slabs, will get xy_dimensions find the correct x and y
        dimensions.
        '''
        basis_list = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis_list, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        supercell.set_cartesian()
        points = supercell.cartesian['coordinates']
        working_volume = surface_creation.select_working_volume(points, 10)
        points = working_volume.points
        xy_dimensions = surface_creation.get_xy_dimensions(points)
        xy_values = xy_dimensions.values.tolist()
        self.assertTrue(xy_values == [[6.0, 21.0], [6.0, 24.0]])

    def OFF_test_get_xy_dimensions_finds_correct_xy_values_111_structure(self):
        '''
        Given some slabs from a 111 reorientated crystal, will get
        xy_dimensions find the correct x and y dimensions.
        '''
        basis_list = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(basis_list, [4, 0, 0], [0, 4, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 14, 14, 14)
        supercell.set_cartesian()
        surface_creation.reorientate_supercell(supercell, [1, 1, 1])
        points = supercell.cartesian['coordinates']
        working_volume = surface_creation.select_working_volume(points, 10)
        points = working_volume.points
        xy_dimensions = surface_creation.get_xy_dimensions(points)
        xy_values = np.around(xy_dimensions.values, 6).tolist()
        expected_xy_values = [[-12.80196, 29.008577], [-12.80196, 29.031442]]
        self.assertTrue(xy_values == expected_xy_values)

    def test_change_dimensions(self):
        '''
        Are the dimensions correctly changed to get the desired next smallest
        point?
        '''
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        dimensions = [0, 3, 0, 3]
        dimensions = surface_creation.change_dimension(dimensions, points, 0)
        self.assertTrue(dimensions == [1, 3, 0, 3])
        dimensions = surface_creation.change_dimension(dimensions, points, 1)
        self.assertTrue(dimensions == [1, 2, 0, 3])
        dimensions = surface_creation.change_dimension(dimensions, points, 2)
        self.assertTrue(dimensions == [1, 2, 1, 3])
        dimensions = surface_creation.change_dimension(dimensions, points, 3)
        self.assertTrue(dimensions == [1, 2, 1, 2])
        surface_creation.change_dimension(dimensions, points, 0)
        self.assertTrue(dimensions == [2, 2, 1, 2])


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from supercell import SuperCell
    from unitcell import UnitCell
    from atom import Atom
    import surface_creation
    import testing_tools
    import linear_algebra as linalg
    import edits
    import crystallography
    unittest.main()
