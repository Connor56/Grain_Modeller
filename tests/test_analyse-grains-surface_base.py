import unittest
import os
import sys
import numpy as np
import pyvista as pv
import pandas as pd


class TestAnalyse_SurfaceBase(unittest.TestCase):

    def OFF_test_PlaneGroup(self):
        '''
        Tests the initialisation of the PlaneGroup class.
        '''
        self.assertTrue(False)

    def OFF_test_Simplex(self):
        '''
        Test the initialisation and the equals part of the Simplex class.
        '''
        self.assertTrue(False)

    def OFF_test_Surface(self):
        '''
        Test the intialisation of the surface class.
        '''
        self.assertTrue(False)

    def test_get_surface(self):
        '''
        Check the correct surface is gained from a set of points.
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cloud = pv.PolyData(array)
        volume = cloud.delaunay_3d(alpha=2)
        shell = volume.extract_geometry()
        points1 = np.array(shell.points[shell.faces[1:4]])
        points2 = np.array(shell.points[shell.faces[5:8]])
        points3 = np.array(shell.points[shell.faces[9:12]])
        points4 = np.array(shell.points[shell.faces[13:16]])
        simplex1 = sb.get_simplex(points1)
        simplex2 = sb.get_simplex(points2)
        simplex3 = sb.get_simplex(points3)
        simplex4 = sb.get_simplex(points4)
        simplexes = [simplex1, simplex2, simplex3, simplex4]
        area = 2.36602540
        volume = np.around(1/6, 8)
        expected_surface = sb.Surface(simplexes, area, volume)
        surface = sb.get_surface(shell)
        self.assertTrue(np.all(surface == expected_surface))

    def test_get_surface_complicated_grain(self):
        '''
        Check the correct surface is still gained from a more a complicated set
        of points.
        '''
        basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                 Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(basis, [3.5, 0, 0], [0, 3.5, 0], [0, 0, 3])
        cuts = [Cut('p', [0.5, 0.5, 1], plane=[1, 1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[-1, 1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[1, -1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[-1, -1, 1])]
        grain = gc.Grain('shell_test', unitcell, cuts, [1, 2, 1])
        supercell = gc.build_grain(grain, 10)
        supercell.set_cartesian()
        points = supercell.cartesian['coordinates']
        cloud = pv.PolyData(points)
        volume = cloud.delaunay_3d(alpha=0)
        shell = volume.extract_geometry()
        surface = sb.get_surface(shell)
        self.assertTrue(len(surface.simplexes) == 1892)
        self.assertTrue(np.around(surface.area, 8) == 5131.0)
        self.assertTrue(np.around(surface.volume, 8) == 20926.0625)

    def test_get_faces(self):
        '''
        Does get_faces return the right index points of the faces in a list.
        '''
        faces = np.array([3, 0, 1, 2, 3, 2, 2, 1, 3, 5, 2, 1, 2, 3, 1])
        faces = sb.get_faces(faces)
        faces = [face.tolist() for face in faces]
        expected_faces = [[0, 1, 2], [2, 2, 1], [5, 2, 1], [3, 1]]
        self.assertTrue(faces == expected_faces)

    def test_get_simplex(self):
        '''
        Check the correct simplex is gained from a set of points.
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cloud = pv.PolyData(array)
        volume = cloud.delaunay_3d(alpha=2)
        shell = volume.extract_geometry()
        points = np.array(shell.points[shell.faces[1:4]])
        simplex = sb.get_simplex(points)
        vectors = np.array([[-1, 1, 0], [-1, 0, 1]]).astype(np.float32)
        area = 0.86602539
        normal = np.array([1, 1, 1]).astype(np.float32)
        normal = normal/np.linalg.norm(normal)
        expected_simplex = sb.Simplex(points, vectors, area, normal)
        self.assertTrue(simplex == expected_simplex)
        self.assertTrue(np.all(simplex.normal == expected_simplex.normal))
        self.assertTrue(np.all(simplex.vectors == expected_simplex.vectors))
        self.assertTrue(np.all(simplex.area == expected_simplex.area))
        self.assertTrue(np.all(simplex.points == expected_simplex.points))

    def test_group_simplexes(self):
        '''
        Check simplexes are correctly grouped by their normals
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cloud = pv.PolyData(array)
        volume = cloud.delaunay_3d()
        shell = volume.extract_geometry()
        surface = sb.get_surface(shell)
        simplexes = surface.simplexes
        grouped_simplexes = sb.group_simplexes(simplexes, 0.1)
        expected_grouped_simplexes = [
            np.array([simplexes[0]]), np.array([simplexes[1]]),
            np.array([simplexes[2]]), np.array([simplexes[3]]),
        ]
        for index in range(len(grouped_simplexes)):
            self.assertTrue(np.all(
                grouped_simplexes[index] == expected_grouped_simplexes[index]))

    def test_group_simplexes_larger_angle(self):
        '''
        Check simplexes are correctly grouped by their normals
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cloud = pv.PolyData(array)
        volume = cloud.delaunay_3d()
        shell = volume.extract_geometry()
        surface = sb.get_surface(shell)
        simplexes = surface.simplexes
        grouped_simplexes = sb.group_simplexes(simplexes, np.pi)
        expected_grouped_simplexes = [
            np.array([simplexes[0], simplexes[1], simplexes[2], simplexes[3]])
        ]
        for index in range(len(grouped_simplexes)):
            self.assertTrue(np.all(
                grouped_simplexes[index] == expected_grouped_simplexes[index]))

    def test_get_areas(self):
        '''
        Check that the area of the grouped simplexes returned in the pandas
        array is what's expected. Can also used simplexes as the description
        of the simplest representation of an N-D unit.
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [2, 0, 0], [1, 0, 1], [0, 0, 2]])
        cloud = pv.PolyData(array)
        volume = cloud.delaunay_3d()
        shell = volume.extract_geometry()
        surface = sb.get_surface(shell)
        surface.grouped_simplexes = sb.group_simplexes(surface.simplexes, 0.01)
        expected_areas = pd.DataFrame({
            'Normal': [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.],
                       [0.5, 1., 0.5]],
            'Area': [1, 2, 1, 1.22474487*2]
        })
        areas = np.around(sb.get_areas(surface.grouped_simplexes), 3)
        expected_areas = np.around(expected_areas, 3)
        self.assertTrue(np.all(areas['Area'] == expected_areas['Area']))
        for index in range(areas['Normal'].shape[0]):
            self.assertTrue(np.all(areas['Normal'].iloc[index] ==
                            expected_areas['Normal'].iloc[index]))

    def test_get_areas_for_a_complicated_surface(self):
        '''
        Checks the group areas algorithm still works correctly when given a
        much more complicated surface to work with.
        '''
        basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                 Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(basis, [3.5, 0, 0], [0, 3.5, 0], [0, 0, 3])
        cuts = [Cut('p', [0.5, 0.5, 1], plane=[1, 1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[-1, 1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[1, -1, 1]),
                Cut('p', [0.5, 0.5, 1], plane=[-1, -1, 1])]
        grain = gc.Grain('shell_test', unitcell, cuts, [1, 2, 1])
        supercell = gc.build_grain(grain, 10)
        supercell.set_cartesian()
        points = supercell.cartesian['coordinates']
        cloud = pv.PolyData(points)
        volume = cloud.delaunay_3d(alpha=0)
        shell = volume.extract_geometry()
        surface = sb.get_surface(shell)
        surface.grouped_simplexes = sb.group_simplexes(surface.simplexes, 0.01)
        areas = sb.get_areas(surface.grouped_simplexes)
        indexes = areas.sort_values(by='Area').index.tolist()
        self.assertTrue(indexes == [8, 7, 0, 6, 5, 4, 3, 1, 2])
        self.assertTrue(areas['Area'].iloc[8] == 2.625)
        self.assertTrue(areas['Area'].iloc[7] == 6.125)

    def test_group_by_plane(self):
        '''
        Test that a group of normals and their areas given in the style
        returned by group_simplexes can be further grouped so that they are
        designated as part of a "Plane Group", which could be something like
        the {111} family of planes, from crystallography, for example.
        '''
        test_normals = pd.DataFrame(
            {'Normal':
                [[1, 1, 1], [5, 2, 1], [0, 0, 1], [0, 0.1, 1]],
             'Area':
                [120, 200, 300, 100]})
        plane_group = [sb.PlaneGroup(np.array([[0, 0, 1]]), '{001}')]
        plane_areas = sb.group_by_plane(test_normals, plane_group, 0.1)
        expected_plane_areas = pd.DataFrame(
            {'Name': ['{001}', 'Remainder'],
             'Plane_Group': [x.planes for x in plane_group]+['None'],
             'Area': [400, 320],
             'Captured_Planes': [[[0, 0, 1], [0, 0.1, 1]],
                                 [[1, 1, 1], [5, 2, 1]]]
             })
        self.assertTrue(np.all(plane_areas == expected_plane_areas))

    def test_group_by_plane_for_multiple_plane_groups(self):
        '''
        Test that the function works for multiple plane groups, and returns
        the expected areas.
        '''
        test_normals = test_normals = pd.DataFrame(
            {'Normal':
                [[1, 1, 1], [920, 920, 1], [-920, 875, 1], [-1, 0, 0],
                 [1, 0, 0], [-24, 0, 0], [0, 1, 0], [0, 0, -2]
                 ],
             'Area':
                [120, 200, 300, 100, 326, 2093, 2389, 0.01]})
        plane_groups = [
            sb.PlaneGroup(
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
                          [0, -1, 0], [0, 0, -1]]),
                '{100}'),
            sb.PlaneGroup(
                np.array([[1, 1, 0], [-1, 1, 0]]),
                '{110}'),
            sb.PlaneGroup(
                np.array([[1, 1, 1]]),
                '{111}')]
        plane_areas = sb.group_by_plane(test_normals, plane_groups, 0.1)
        expected_plane_areas = pd.DataFrame(
            {'Name': ['{100}', '{110}', '{111}', 'Remainder'],
             'Plane_Group': [x.planes for x in plane_groups]+['None'],
             'Area': [4908.01, 500, 120, 0],
             'Captured_Planes': [
                [[-1, 0, 0], [1, 0, 0], [-24, 0, 0], [0, 1, 0], [0, 0, -2]],
                [[920, 920, 1], [-920, 875, 1]],
                [[1, 1, 1]],
                'None']
             })
        self.assertTrue(np.all(plane_areas == expected_plane_areas))

    def test_group_by_plane_for_multiple_plane_groups_large_angle(self):
        '''
        Test that the function works for multiple plane groups, with a very
        large angle. Shouldn't return a value over the total area, as each
        plane should only be counted in a single plane group.
        '''
        areas = [120, 200, 300, 100, 326, 2093, 2389, 0.01]
        test_normals = test_normals = pd.DataFrame(
            {'Normal':
                [[1, 1, 1], [920, 920, 1], [-920, 875, 1], [-1, 0, 0],
                 [1, 0, 0], [-24, 0, 0], [0, 1, 0], [0, 0, -2]
                 ],
             'Area': areas})
        plane_groups = [
            sb.PlaneGroup(
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
                          [0, -1, 0], [0, 0, -1]]),
                '{100}'),
            sb.PlaneGroup(
                np.array([[1, 1, 0], [-1, 1, 0]]),
                '{110}'),
            sb.PlaneGroup(
                np.array([[1, 1, 1]]),
                '{111}')]
        plane_areas = sb.group_by_plane(test_normals, plane_groups, 3)
        #print(plane_areas)
        self.assertTrue(np.sum(plane_areas['Area'].iloc[:-1] <= np.sum(areas)))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.grains import surface_base as sb
    import grain_creation as gc
    from edits import Cut
    from atom import Atom
    from unitcell import UnitCell
    unittest.main()
