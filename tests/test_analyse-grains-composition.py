import unittest
import os
import sys


class TestAnalyse_Composition(unittest.TestCase):

    def test_is_identical(self):
        '''
        Does is_composition_identical correctly identify when two files have
        the exact same number of the different atoms types?
        '''
        files = ['test_files/Truncated_Octahedron_Major_1000_atoms.in',
                 'test_files/Truncated_Octahedron_Major_11000_atoms.in',
                 'test_files/Truncated_Octahedron_Minor_11000_atoms.in']
        style = 'LAMMPS'
        self.assertTrue(composition.is_identical(style, files[1:]))
        self.assertFalse(composition.is_identical(style, files))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.grains import composition
    unittest.main()
