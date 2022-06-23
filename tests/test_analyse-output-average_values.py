import unittest
import os
import sys
import numpy as np


class TestFastGrainSupercell(unittest.TestCase):

    def test_get_file_data(self):
        '''
        Test the style definition correctly selects desired columns.
        '''
        file = './test_files/average_atom_finder_test.xyz'
        style = 'Lammps_Grain_Positions'
        atom_values = average_values.get_file_data(file, style)
        self.assertTrue(len(atom_values) == 1)
        self.assertTrue(len(atom_values[0]) == 8531)
        self.assertTrue(
            (np.array([len(list) for list in atom_values[0]]) == 5).all())

    def test_get_average_positions(self):
        '''
        Test function correctly gets the average position of atoms in an
        output file.
        '''
        file = './test_files/average_atom_position_large_test.xyz'
        style = 'Lammps_Grain_Positions'
        average_positions = (
            average_values.get_average_positions(file, style, 0))
        self.assertTrue(len(average_positions) == 8531)
        list = np.round(average_positions[1], 6).tolist()
        self.assertTrue(list == [1.0, 0.097935, 0.307663, 0.375276])

    def test_get_average_positions_2(self):
        '''
        Test function correctly averages position data by using a small file.
        '''
        file = './test_files/average_atom_small_test.xyz'
        style = 'Lammps_Grain_Positions'
        average_positions = (
            average_values.get_average_positions(file, style, 0))
        self.assertTrue(np.round(average_positions[1], 4).tolist()
                        == [2.0, 4.55, 5.7, 9.0])
        self.assertTrue(np.round(average_positions[2], 4).tolist()
                        == [1.0, 5.5, 7.1, 8.05])

    def test_create_file_from_averaged_lammps_data(self):
        '''
        Test a lammps data input file is correctly created from averaged_data.
        '''
        averaged_data = {1: np.array([1., 1.0, 1., 1.]),
                         4: np.array([2., 3., 5., 6.])}
        average_values.create_file_from_averaged_data_lammps(
            averaged_data, 'test')
        with open('test.in', 'r') as file:
            data = file.read().split('\n')
        os.system('rm test.in')
        expected_data = [
            'test', '', '2 atoms', '2 atom types', '1.0 3.0 xlo xhi',
            '1.0 5.0 ylo yhi', '1.0 6.0 zlo zhi', '0 0 0 xy xz yz', '',
            'Atoms', '', '1 1 1.0 1.0 1.0', '4 2 3.0 5.0 6.0']
        self.assertTrue(data == expected_data)


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.output import average_values
    unittest.main()
