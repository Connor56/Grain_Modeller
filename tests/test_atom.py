import os
import sys
import unittest
import numpy as np

from copy import copy

class TestAtom(unittest.TestCase):

    def test_atom_instantiation(self):
        '''
        Test atom instantiates correctly.
        '''
        test_atom = atom.Atom('Pt', 0, 0, 0)
        self.assertTrue(repr(test_atom) == "Atom('Pt', 0.0, 0.0, 0.0)")
        self.assertRaisesRegex(
            ValueError, 'Element must be a string.',
            atom.Atom, 1, 2, 3, 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, x is not.',
            atom.Atom, 'Pt', 'Fe', 3, 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, y is not.',
            atom.Atom, 'Pt', 1, 'Fe', 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, z is not.',
            atom.Atom, 'Pt', 2, 3, 'Za')

    def test_atom_instantiation_2(self):
        '''
        Test atom instantiates correctly with float values.
        '''
        test_atom = atom.Atom('Pt', 0.5, 0.5, 0.5)
        self.assertTrue(repr(test_atom) == "Atom('Pt', 0.5, 0.5, 0.5)")
        self.assertRaisesRegex(
            ValueError, 'Element must be a string.',
            atom.Atom, 1, 2, 3, 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, x is not.',
            atom.Atom, 'Pt', 'Fe', 3, 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, y is not.',
            atom.Atom, 'Pt', 1, 'Fe', 4)
        self.assertRaisesRegex(
            ValueError, 'All coordinates must be numeric, z is not.',
            atom.Atom, 'Pt', 2, 3, 'Za')

    def test__copy__(self):
        '''
        Test copy returns an exact copy of the atom when using the copy module,
        that doesn't reference the same object.
        '''
        test_atom = atom.Atom('Pt', 0, 0, 0)
        atom_copy = copy(test_atom)
        self.assertFalse(test_atom == atom_copy)
        self.assertTrue(repr(test_atom) == repr(atom_copy))

    def test_set_cartesian(self):
        '''
        Test a given vector space creates the correct cartesian coordinates.
        '''
        test_atom = atom.Atom('Fe', 0.4, 0.2, 0.5)
        vector_space = np.array([[1, 2, 4], [0, 2, 0], [0, 0, 3]]).T
        test_atom.set_cartesian(vector_space)
        cartesian = np.around(test_atom.cartesian['coordinates'][0], 6)
        self.assertTrue(cartesian.tolist() == [0.4, 1.2, 3.1])






if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import atom
    unittest.main()
