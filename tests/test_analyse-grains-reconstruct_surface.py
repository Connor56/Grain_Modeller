import unittest
import os
import sys
import numpy as np


class TestAnalyse_ReconstructSurface(unittest.TestCase):

    def test_analyse(self):
        '''
        Does analyse surface box get the correct information for a shape with
        a known surface and size?
        '''
        basis = [Atom('Fe', 0, 0, 0), Atom('Fe', 0.5, 0.5, 0),
                 Atom('Pt', 0.5, 0, 0.5), Atom('Pt', 0, 0.5, 0.5)]
        unitcell = UnitCell(basis, [3.82, 0, 0], [0, 3.82, 0], [0, 0, 3.71])
        supercell = SuperCell(unitcell, 20, 20, 20)
        supercell.set_cartesian()
        box = [[10, 50], [10, 50], [10, 50]]
        atoms = supercell.cartesian['coordinates']
        surface = rs.analyse(atoms, box=box, angle=0.1)
        clipped_areas = surface.grouped_area.sort_values(
            by='Area', ascending=False)
        clipped_areas = clipped_areas[clipped_areas['Area'] > surface.area/100]
        normals = clipped_areas['Normal']
        self.assertTrue(np.all(
            np.isclose(normals.iloc[0], [-0.009, -0.032, -1], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[1], [-1, -0.009, -0.021], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[3], [-0.0065, -0.021, 1], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[5], [1.0, -0.007, -0.012], atol=0.001)))

    def test_analyse_large_complicated_grain(self):
        '''
        Does analyse surface box get the correct information for a shape with
        a complex morphology and a twin in the middle?
        '''
        atoms = utility.read_atom_file("test_files/smfe12_small_grain.in")
        box = [[51, 121], [73, 143], [65, 135]]
        surface = rs.analyse(atoms, box=box, angle=0.1)
        clipped_areas = surface.grouped_area.sort_values(
            by='Area', ascending=False)
        clipped_areas = clipped_areas[clipped_areas['Area'] > surface.area/100]
        normals = clipped_areas['Normal']
        self.assertTrue(np.all(
            np.isclose(normals.iloc[0], [1, 0.033, 0.0070], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[1], [-1, 0.034, -0.084], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[3], [-1, -0.998, -0.0013], atol=0.001)))
        self.assertTrue(np.all(
            np.isclose(normals.iloc[6], [-0.0048, -0.665, 1], atol=0.001)))

    def test_surface_atoms(self):
        '''
        Does surface_atoms correctly return the surface atoms of a given point
        list, representing a crystal structure.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        supercell = SuperCell(unitcell, 40, 40, 40)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[10, 30], [10, 30], [10, 30]]
        surface_atoms = rs.surface_atoms(atoms, box)
        expected_atoms = (38*38*6)+(38*12)+8
        self.assertTrue(surface_atoms.shape[0] == expected_atoms)

    def test_surface_atoms_more_complex_unitcell(self):
        '''
        Does surface_atoms correctly return the surface atoms of a given point
        list, representing a crystal structure.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [2, 0, 0], [0, 2.5, 0], [0, 0, 2])
        supercell = SuperCell(unitcell, 40, 40, 40)
        supercell.set_cartesian()
        atoms = supercell.cartesian['coordinates']
        box = [[10, 30], [10, 30], [10, 30]]
        surface_atoms = rs.surface_atoms(atoms, box, 12)
        expected_atoms = (38*38*6)+(38*12)+8
        self.assertTrue(surface_atoms.shape[0] == expected_atoms)

    def test_surface_atoms_for_a_complex_grain(self):
        '''
        Does surface_atoms correctly return the surface atoms of a given point
        list, representing a crystal structure.
        '''
        atoms = fr.lammps_file(
            'test_files/Truncated_Octahedron_Major_11000_atoms.in')[:, 1:]
        box = [[30, 45], [30, 45], [30, 42]]
        surface_atoms = rs.surface_atoms(atoms, box, 12)
        self.assertTrue(surface_atoms.shape[0] == 2123)

    def test_surface_atoms_for_a_complex_grain_with_a_twin(self):
        '''
        Does surface_atoms correctly return the surface atoms of a large grain
        with a twin structure inside of it?
        '''
        atoms = utility.read_atom_file("test_files/smfe12_small_grain.in")
        box = [[51, 121], [73, 143], [65, 135]]
        surface_atoms = rs.surface_atoms(atoms, box, 12)
        self.assertTrue(surface_atoms.shape[0] == 21835)


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    from analyse.grains import reconstruct_surface as rs
    import utility
    import file_reader as fr
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    unittest.main()
