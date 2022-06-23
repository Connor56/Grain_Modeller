import unittest
import os
import sys
import numpy as np


class TestCartesianEdits(unittest.TestCase):

    def test_box_select(self):
        '''
        Does box select return an accruate truth array, for out=False and
        out=True.
        '''
        atoms = np.array([[1, 3, 4.3], [2, 5, 5], [2, 1, 2]])
        box = [[0.9, 1.1], [2.9, 3.1], [4.2, 4.4]]
        truth_array = ce.box_select(atoms, box, out=False)
        expected_truth_array = [True, False, False]
        self.assertTrue(np.all(expected_truth_array == truth_array))
        # For out=True
        truth_array = ce.box_select(atoms, box, out=True)
        expected_truth_array = [False, True, True]
        self.assertTrue(np.all(expected_truth_array == truth_array))

    def test_box_select_complicated(self):
        '''
        Test box select works for a larger array and bigger box.
        '''
        atoms = np.array(
            [[1, 3, 4.3], [2, 5, 5], [2, 1, 2], [-1, 3, 2], [2, 2.4, 1.5],
             [24, -24, 23], [-2, -2, -53], [25, 3, -2.5], [np.pi, 5.1, 7]])
        box = [[-1, 4], [3, 35], [4, 8]]
        truth_array = ce.box_select(atoms, box, out=False)
        expected_truth_array = [
            False, True, False, False, False, False, False, False, True]
        self.assertTrue(np.all(expected_truth_array == truth_array))
        # For out=True
        truth_array = ce.box_select(atoms, box, out=True)
        self.assertTrue(np.all(truth_array == np.invert(expected_truth_array)))

    def test_box_select_large_atom_array(self):
        '''
        Test box select on a large atom_array.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 40, 40, 40)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[20, 25], [20, 50], [20, 25]]
        box_mask = ce.box_select(atoms, box)
        box_atoms = atoms[box_mask]
        self.assertTrue(np.all(box_atoms[:, 0] > 20))
        self.assertTrue(np.all(box_atoms[:, 0] < 25))
        self.assertTrue(np.all(box_atoms[:, 1] > 20))
        self.assertTrue(np.all(box_atoms[:, 1] < 50))
        self.assertTrue(np.all(box_atoms[:, 2] > 20))
        self.assertTrue(np.all(box_atoms[:, 2] < 25))

    def test_box_select_warns_user_when_no_atoms_selected(self):
        '''
        Does the algorithm correctly warn users when no atoms are selected?
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 10, 10, 10)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[-40, -10], [-40, -10], [-40, -10]]
        self.assertWarnsRegex(
            UserWarning,
            "No atoms within box:\n\\[\\[\\-40 \\-10\\]\n \\[\\-40 \\-10\\]\n "
            "\\[\\-40 \\-10\\]\\]\nCheck box dimensions if this was not "
            "deliberate.",
            ce.box_select, atoms, box)

    def test_box_select_raises_error_when_box_dimensions_are_incorrect(self):
        '''
        Does box select throw a valueError when the box dimensions aren't those
        expected?
        '''
        atoms = np.array([[1, 1, 1]])
        box = [[1, 2, 3], [1, 2, 3]]
        self.assertRaisesRegex(
            ValueError,
            r"Box dimesions are: (2, 3), the required dimensions are (3, 2). "
            r"Please read the box_select description for more details."
        )



if __name__ == '__main__':
    current_directory = os.getcwd()
    folder_name = 'grain_modeller'
    package_directory_index = current_directory.index(folder_name)
    package_directory_index += len(folder_name)
    package_directory = current_directory[:package_directory_index]
    sys.path.append(package_directory+'/grain_modeller')
    import cartesian_edits as ce
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    unittest.main()
