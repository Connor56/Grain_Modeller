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
        expected_indexes = [10, 11, 13, 8, 12, 5, 9, 4, 0, 1, 7, 6, 3, 2]
        self.assertTrue(indexes == expected_indexes)

    def test_get_shell(self):
        '''
        Does it get the expected surface information?
        '''
        array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        shell = ds.get_shell(array, 2)
        expected_surf_norms = np.array(
            [[1, 1, 1], [-1, 0 ,0], [0, -1, 0], [0, 0, -1]])
        norms = np.linalg.norm(expected_surf_norms, axis=1).reshape(4, -1)
        expected_surf_norms = expected_surf_norms/norms
        surf_norms = np.array(shell.face_normals)
        truth_array = np.isclose(expected_surf_norms, surf_norms, atol=0.0001)
        self.assertTrue(np.all(truth_array))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import analyse.grains.delaunay_surface as ds
    import utility
    unittest.main()
