import unittest
import time
import os
import sys
from itertools import combinations
import numpy as np
import os


class TestUtility(unittest.TestCase):

    def test_get_best_normal(self):
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                            [1, -1, 1]])
        centre = np.array([4, 4 ,4])
        point = np.array([6, 4.0009, 8])
        best_normal = utility.get_best_normal(normals, centre, point)
        self.assertTrue(best_normal.tolist() == [1, -1, 1])

    def test_order_difference(self):
        '''
        Does it correctly caculate the difference in order between two easy
        and two slightly more complicated values.
        '''
        order_difference = utility.order_difference(1000, 0.1)
        self.assertTrue(np.around(order_difference, 1) == 4)
        order_difference = utility.order_difference(1200, 0.5)
        self.assertTrue(np.around(order_difference, 4) == 3.3802)
        order_difference = utility.order_difference(16, 0.25, 2)
        self.assertTrue(np.around(order_difference, 1) == 6)

    def test_read_atom_file(self):
        with open('test.in', 'w') as file:
            file.write('Atoms\n\n1 This 1 1 1\n2 is 2 2 2\n3 a 3 3 3\n'
                        + '4 test 4 4 4\n\nMasses')
        array = utility.read_atom_file('test.in')
        os.remove('test.in')
        expected_array = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        self.assertTrue(np.all(array == expected_array))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import utility
    unittest.main()
