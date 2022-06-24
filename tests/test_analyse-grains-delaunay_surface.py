import unittest
import os
import sys
import numpy as np


class TestAnalyse_DelaunaySurface(unittest.TestCase):

    def test_analyse_surface(self):
        '''
        Check analyse surface gives the correct information for a shape with a
        known surface and size?
        '''
        atoms = utility.read_atom_file(
            "test_files/test_grain_analysis/"
            "test_analyse_surface_Cuboid_3000_atoms.in")
        surface = ds.analyse(atoms, alpha=2)
        indexes = surface.grouped_area.sort_values(by='Area').index.tolist()
        normals = surface.grouped_area.sort_values(by='Area')['Normal']
        self.assertTrue(normals.iloc[-1].tolist() == [0, 0, 1])
        self.assertTrue(normals.iloc[-2].tolist() == [0, 0, -1])
        self.assertTrue(normals.iloc[-3].tolist() == [1, 0, 0])
        self.assertTrue(normals.iloc[-6].tolist() == [0, -1, 0])

    def test_get_shell(self):
        '''
        Does it get the expected surface information?
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        shell = ds.get_shell(array, 2)
        expected_surface_normals = np.array(
            [[1, 1, 1], [-1, 0 ,0], [0, -1, 0], [0, 0, -1]])
        norms = np.linalg.norm(expected_surface_normals, axis=1).reshape(4, -1)
        expected_surface_normals = expected_surface_normals/norms
        expected_surface_normals = expected_surface_normals[
            np.lexsort(expected_surface_normals.T[::-1])]
        surface_normals = np.array(shell.face_normals)
        surface_normals = surface_normals[np.lexsort(surface_normals.T[::-1])]
        truth_array = np.isclose(
            expected_surface_normals, surface_normals, atol=0.0001)
        self.assertTrue(np.all(truth_array))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import analyse.grains.delaunay_surface as ds
    import utility
    unittest.main()
