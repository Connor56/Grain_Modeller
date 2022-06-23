import unittest
import os
import sys
import numpy as np
import cProfile
import pstats
from pstats import SortKey
import timeit
import time


class TestSuperCell(unittest.TestCase):

    def test_supercell_instantiation(self):
        '''
        Check supercell instantiates correctly.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        self.assertTrue(
            repr(supercell) == "SuperCell(UnitCell([Atom('Fe', 0.0, 0.0, 0.0),"
            + " Atom('Pt', 0.5, 0.5, 0.5)], [3, 0, 0], [0, 3, 0], [0, 0, 3]), "
            + "10, 10, 10)")

    def test_large_supercell_instantiation(self):
        '''
        Check a large supercell can be instantiated correctly.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0.0),
                      Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(
            test_basis, [3.83, 0, 0], [0, 3.83, 0], [0, 0, 3.711])
        time1 = time.time()
        SuperCell(unitcell, 60, 60, 60)
        time_run = time.time()-time1
        self.assertTrue(time_run < 1)

    def test_repeat_atoms(self):
        '''
        Check the repeat atoms method repeats atoms correctly to give a
        supercell.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 2, 1, 1)
        self.assertTrue(
            np.all(supercell.fractional ==
            np.array([('Fe', [0.0, 0.0, 0.0]), ('Fe', [1.0, 0.0, 0.0]),
                      ('Pt', [0.5, 0.5, 0.5]), ('Pt', [1.5, 0.5, 0.5])],
                      dtype=[('element', 'U10'), ('coordinates', 'f8', 3)])))

    def test_randomise_correctly_randomises_the_structure(self):
        '''
        Test the supercell randomise function significantly changes the
        positions in the atom list.
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        test_unitcell = UnitCell(
            test_basis, [3.83, 0, 0], [0, 3.83, 0], [0, 0, 3.711])
        test_supercell = SuperCell(test_unitcell, 10, 10, 10)
        atoms = test_supercell.fractional['element'].copy()
        test_supercell.randomise([0.5, 0.5])
        randomised_atoms = test_supercell.fractional['element']
        fe_ratio = np.sum(randomised_atoms == 'Fe')/randomised_atoms.shape[0]
        self.assertTrue(np.isclose(0.5, fe_ratio, rtol=0, atol=0.02))
        test_supercell.randomise([0.9, 0.1])
        randomised_atoms = test_supercell.fractional['element']
        fe_ratio = np.sum(randomised_atoms == 'Fe')/randomised_atoms.shape[0]
        self.assertTrue(np.isclose(0.9, fe_ratio, rtol=0, atol=0.02))

    def test_set_cartesian(self):
        '''
        Does set cartesian turn the fractional coordinates into cartesian ones
        correctly?
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0)]
        test_unitcell = UnitCell(
            test_basis, [3.5, 0, 0], [0, 3., 0], [0, 0, 4.7])
        test_supercell = SuperCell(test_unitcell, 1, 1, 2)
        test_supercell.set_cartesian()
        atoms = test_supercell.cartesian
        expected_atoms = [[0, 0, 0], [0, 0, 4.7],
                          [1.75, 1.5, 0], [1.75, 1.5, 4.7]]
        self.assertTrue(atoms['coordinates'].tolist() == expected_atoms)
        test_unitcell = UnitCell(
            test_basis, [3.5, 1, 0.2], [1.2, 3., 0.4], [2.11, 1.2, 4.7])
        test_supercell = SuperCell(test_unitcell, 1, 1, 1)
        test_supercell.set_cartesian()
        atoms = np.around(test_supercell.cartesian['coordinates'], 6)
        expected_atoms = [[0.0, 0.0, 0.0], [2.821252, 1.285706, 0.0]]
        self.assertTrue(atoms.tolist() == expected_atoms)

    def test_set_fractional(self):
        '''
        Does set_fractional turn cartesian coordinates into fractional
        coordinates as expected? This is important for performing operations
        such as reflections.
        '''
        # Check it works for a complicated vector space and simple coordinates
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0)]
        test_unitcell = UnitCell(
            test_basis, [3.5, 0, 0], [1, 3., 0], [1, 2.9, 4.7])
        test_supercell = SuperCell(test_unitcell, 1, 1, 1)
        test_supercell.set_cartesian()
        test_supercell.cartesian['coordinates'] = np.array(
            [[3.5, 0, 0], [2, 5.8, 9.4]])
        test_supercell.set_fractional()
        expected_coordinates = np.array([[1, 0, 0], [0, 0, 2]])
        coordinates = test_supercell.fractional['coordinates']
        self.assertTrue(
            np.all(np.isclose(expected_coordinates, coordinates, atol=1e-8)))

        # Check it works for a complicated vector space and complex coordinates
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0)]
        test_unitcell = UnitCell(
            test_basis, [3.5, 0, 0], [-1, 3., 0], [1.4, -2.91, 4.7])
        test_supercell = SuperCell(test_unitcell, 1, 1, 1)
        test_supercell.set_cartesian()
        test_supercell.cartesian['coordinates'] = np.array(
            [[3.5, 4.3, 1], [32, -1, -9.4]])
        test_supercell.set_fractional()
        expected_coordinates = np.array(
            [[1.38338399, 1.63971631, 0.21276596],
             [9.29333333, -2.27333333, -2]])
        coordinates = test_supercell.fractional['coordinates']
        self.assertTrue(
            np.all(np.isclose(expected_coordinates, coordinates, atol=1e-8)))














if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from supercell import SuperCell
    from unitcell import UnitCell
    from atom import Atom
    unittest.main()
