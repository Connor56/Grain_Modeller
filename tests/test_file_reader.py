import unittest
import os
import sys
import pandas as pd
import numpy as np


class TestFileReader(unittest.TestCase):

    def test_xyz_file(self):
        '''
        Does it correctly read in a file of the xyz file format?
        '''
        file_string = (
            "ITEM: TIMESTEP\n10\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS ff ff ff\n1 9\n1 9\n1 9\n"
            "ITEM: ATOMS id type test_1 test_2\n"
            "1 1 -1 -5\n"
            "2 1 -3 3\n"
            "3 1 -8 7\n")
        with open('test.xyz', 'w') as f:
            f.write(file_string)
        data = file_reader.xyz_file('test.xyz')
        step_data = data.steps[0]
        expected_atom_data_1 = pd.DataFrame(
            {'id': [1., 2., 3.],
             'type': [1., 1., 1.],
             'test_1': [-1., -3., -8.],
             'test_2': [-5., 3., 7.]})
        expected_box_data_1 = np.array([[1., 9.], [1., 9.], [1., 9.]])
        self.assertTrue(np.all(step_data.atom_data == expected_atom_data_1))
        self.assertTrue(step_data.number_of_atoms == 3)
        self.assertTrue(step_data.time_step == 10)
        self.assertTrue(np.all(step_data.box_data == expected_box_data_1))
        os.remove('test.xyz')

        # Repeat test but for multiple step entries
        file_string = (
            "ITEM: TIMESTEP\n10\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS ff ff ff\n1 9\n1 9\n1 9\n"
            "ITEM: ATOMS id type test_1 test_2\n"
            "1 1 -1 -5\n"
            "2 1 -3 3\n"
            "3 1 -8 7\n"
            "ITEM: TIMESTEP\n30\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS ff ff ff\n1 10\n1 10\n-1 12\n"
            "ITEM: ATOMS id type test_1 test_2\n"
            "1 1 -1 -8\n"
            "2 1 1 10\n"
            "3 1 -2 -3\n")
        with open('test.xyz', 'w') as f:
            f.write(file_string)
        data = file_reader.xyz_file('test.xyz')
        step_data = data.steps
        expected_atom_data_2 = pd.DataFrame(
            {'id': [1., 2., 3.],
             'type': [1., 1., 1.],
             'test_1': [-1., 1., -2.],
             'test_2': [-8., 10., -3.]})
        expected_box_data_2 = np.array([[1., 10.], [1., 10.], [-1., 12.]])
        self.assertTrue(np.all(step_data[0].atom_data == expected_atom_data_1))
        self.assertTrue(np.all(step_data[1].atom_data == expected_atom_data_2))
        self.assertTrue(step_data[1].time_step == 30)
        self.assertFalse(step_data[0] == step_data[1])
        self.assertTrue(np.all(step_data[1].box_data == expected_box_data_2))
        self.assertTrue(data.name == 'test')
        self.assertTrue(data.type == '.xyz')
        self.assertTrue(len(data.steps) == 2)
        os.remove('test.xyz')

    def test_index_matches(self):
        '''
        Does index regex matches return the correct indexes from a given string
        list?
        '''
        regex = r"this is \w+ test"
        string_list = ['this is a test', 'this is b test', 'this is test',
                       'this is very test', 'this is a naughty test']
        indexes = file_reader.index_matches(string_list, regex)
        expected_indexes = [0, 1, 3]
        self.assertTrue(indexes == expected_indexes)
        string_list = string_list[2:5:3]
        self.assertRaises(
            ValueError, file_reader.index_matches, string_list, regex)

    def test_cif_file(self):
        '''
        Will cif_file correctly read information from a cif file into a
        supercell?
        '''
        self.assertTrue(False)

    def test_lammps_file(self):
        '''
        Does lammps_file correctly read in a LAMMPS data input file?
        '''
        file_string = (
            "name_of_file\n\n"
            "3 atoms\n"
            "2 atom types\n"
            "-10.0 10.0 xlo xhi\n"
            "-10.5 11.0 ylo yhi\n"
            "-10.232 12.5 zlo zhi\n"
            "0 0 0 xy xz yz\n\n"
            "Atoms\n\n"
            "1 1 1 3 5\n"
            "2 2 9.1 3.82 4.1\n"
            "3 2 -10 -10.5 -10.232\n\n"
            "Masses\n\n"
            "1 55.85"
            "2 195.08")
        file_name = 'lammps_test_file.in'
        with open(file_name, 'w') as f:
            f.write(file_string)
        array = file_reader.lammps_file(file_name)
        expected_array = np.array(
            [[1, 1, 3, 5], [2, 9.1, 3.82, 4.1], [2, -10, -10.5, -10.232]])
        self.assertTrue(np.all(array == expected_array))
        os.remove(file_name)



if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import file_reader
    unittest.main()
