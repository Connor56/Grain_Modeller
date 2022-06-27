import unittest
import time
import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import cProfile, pstats
import copy


class TestGrainCreation(unittest.TestCase):

    def test_Grain_class_instantiation(self):
        cut_list = [[[1, 1, 1], [0, 0, 1]], [[1, 0, 1], [1, 0, 0]]]
        name = 'test'
        repeat_ratio = [1, 1, 1]
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell = UnitCell(
            test_basis, [3.82, 0, 0], [0, 3.82, 0], [0, 0, 3.711])
        grain = gc.Grain(name, unitcell, cut_list, repeat_ratio)
        expected_representation = (f"Grain('test', {unitcell}, {cut_list}, "
                                  + f"{repeat_ratio})")
        self.assertTrue(repr(grain) == expected_representation)

    def test_compositionally_match(self):
        '''
        Can the composition of similar grains be correctly matched?
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3.7])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grains = [gc.Grain('test', unitcell, cuts, repeat_ratio)]
        grains = gc.scale_grains(grains, 2000)
        self.assertTrue(grains[0].supercell.fractional.shape[0] == 2200)
        cuts_2 = [Cut('p', [1, 1, 0.5], plane=[1, 1, 1]),
                  Cut('p', [0, 1, 2], plane=[0, 0, 0.5])]
        repeat_ratio_2 = [1, 2, 1]
        grains = [gc.Grain('test', unitcell, cuts, repeat_ratio),
                  gc.Grain('test_2', unitcell, cuts_2, repeat_ratio_2)]
        grains = gc.compositionally_match(grains, 5000)
        sizes = [grain.supercell.fractional.shape[0] for grain in grains]
        compositions = [gc.get_composition(grain).values for grain in grains]
        compositions = [comp.tolist()[0] for comp in compositions]
        self.assertTrue(np.all(np.array(sizes) == 5000))
        self.assertTrue(compositions[0] == compositions[1])

    def test_compositionally_match_2(self):
        '''
        Does the function correctly match three grain types at various atom
        target sizes?
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell = UnitCell(
            test_basis, [3.83, 0, 0], [0, 3.83, 0], [0, 0, 3.711])
        cuts_1 = [Cut('p', [1, 0, 0], plane=[1, 1, 1]),
                  Cut('p', [1, 0, 0], plane=[1, -1, 1]),
                  Cut('p', [1, 0, 0], plane=[1, 1, -1]),
                  Cut('p', [1, 0, 0], plane=[1, -1, -1]),
                  Cut('p', [-1, 0, 0], plane=[-1, 1, 1]),
                  Cut('p', [-1, 0, 0], plane=[-1, 1, -1]),
                  Cut('p', [-1, 0, 0], plane=[-1, -1, 1]),
                  Cut('p', [-1, 0, 0], plane=[-1, -1, -1])]
        cuts_2, cuts_3 = copy.deepcopy(cuts_1), copy.deepcopy(cuts_1)
        for cut in cuts_2:
            cut.point = cut.point*1.4
        for cut in cuts_3:
            cut.point = cut.point*1.8
        print('It begins here:')
        repeat_ratio = [1, 1, 1]
        grains = [
            gc.Grain('Octahedron', unitcell, cuts_1, repeat_ratio),
            gc.Grain('Truncated_Octahedron_Minor', unitcell, cuts_2,
                     repeat_ratio),
            gc.Grain('Truncated_Octahedron_Major', unitcell, cuts_3,
                     repeat_ratio)]
        # Check for a 10,000 Atom Target
        grains_1 = gc.compositionally_match(grains, 10000)
        sizes = [grain.supercell.fractional.shape[0] for grain in grains]
        compositions = [gc.get_composition(grain).values for grain in grains]
        compositions = [comp.tolist()[0] for comp in compositions]
        self.assertTrue(np.all(np.array(sizes) == 10000))
        self.assertTrue(compositions[0] == compositions[1] == compositions[2])
        # Check for a 30,000 Atom Target
        grains_1 = gc.compositionally_match(grains, 30000)
        sizes = [grain.supercell.fractional.shape[0] for grain in grains]
        compositions = [gc.get_composition(grain).values for grain in grains]
        compositions = [comp.tolist()[0] for comp in compositions]
        self.assertTrue(np.all(np.array(sizes) == 30000))
        self.assertTrue(compositions[0] == compositions[1] == compositions[2])
        # Check for a 60,000 Atom Target
        grains_1 = gc.compositionally_match(grains, 60000)
        sizes = [grain.supercell.fractional.shape[0] for grain in grains]
        compositions = [gc.get_composition(grain).values for grain in grains]
        compositions = [comp.tolist()[0] for comp in compositions]
        self.assertTrue(np.all(np.array(sizes) == 60000))
        self.assertTrue(compositions[0] == compositions[1] == compositions[2])

    def test_scale_grains(self):
        '''
        Test if the size grains produces grains as close as possible to the
        desired atom target.
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3.7])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grains = [gc.Grain('test', unitcell, cuts, repeat_ratio)]
        grains = gc.scale_grains(grains, 2000)
        self.assertTrue(grains[0].supercell.fractional.shape[0] == 2200)
        cuts_2 = [Cut('p', [1, 1, 0.5], plane=[1, 1, 1]),
                  Cut('p', [0, 1, 2], plane=[0, 0, 0.5])]
        repeat_ratio_2 = [1, 2, 1]
        grains = [gc.Grain('test', unitcell, cuts, repeat_ratio),
                  gc.Grain('test_2', unitcell, cuts_2, repeat_ratio_2)]
        grains = gc.scale_grains(grains, 5000)
        self.assertTrue(np.isclose(grains[0].supercell.fractional.shape[0],
                        5000, atol=1000))
        self.assertTrue(np.isclose(grains[1].supercell.fractional.shape[0],
                        5000, atol=1000))

    def test_build_grain(self):
        '''
        Test that build_grain produces a grain shape based on a Grain object
        correctly.
        '''
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        supercell = gc.build_grain(grain, 10)
        atoms = supercell.fractional['coordinates']
        self.assertTrue(np.max(atoms[:, 2]) <= 5)

    def test_alter_cuts_plane(self):
        '''
        Test that alter cuts correctly resizes cut positions to match a given
        size factor multiple.
        '''
        cuts = [Cut('p', [1, 1, 0], plane=[0, 0, 1]),
                Cut('p', [1, 2, 1], plane=[1, 1, 1])]
        repeats = [1, 1, 1]
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        grain = gc.Grain('test', unitcell, cuts, repeats)
        cuts = gc.alter_cuts(8, grain)
        cuts = [[cut.plane.tolist(), cut.point.tolist()] for cut in cuts]
        expected_cuts = [[[0, 0, 1], [8, 8, 0]], [[1, 1, 1], [8, 16, 8]]]
        self.assertTrue(cuts == expected_cuts)
        self.assertFalse(grain.cuts == expected_cuts)
        cuts = [Cut('p', [1, 1, 0], plane=[0, 0, 1]),
                Cut('p', [1, 2, 1], plane=[1, 1, 1])]
        repeats = [1, 5, 2]
        grain = gc.Grain('test', unitcell, cuts, repeats)
        cuts = gc.alter_cuts(13, grain)
        cuts = [[cut.plane.tolist(), cut.point.tolist()] for cut in cuts]
        expected_cuts = [[[0, 0, 1], [13, 65, 0]], [[1, 1, 1], [13, 130, 26]]]
        self.assertTrue(cuts == expected_cuts)
        self.assertFalse(grain.cuts == expected_cuts)

    def test_alter_cuts_spherical(self):
        '''
        Test that alter cuts correctly resizes cut positions and radius of
        spherical cuts.
        '''
        cuts = [Cut('s', [1, 1, 0], radius=3),
                Cut('s', [1, 2, 1], radius=4)]
        repeats = [1, 1, 1]
        basis = [Atom('Fe', 0, 0, 0)]
        unitcell = UnitCell(basis, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        grain = gc.Grain('test', unitcell, cuts, repeats)
        cuts = gc.alter_cuts(8, grain)
        cuts = [[cut.radius, cut.point.tolist()] for cut in cuts]
        expected_cuts = [[24, [8, 8, 0]], [32, [8, 16, 8]]]
        self.assertTrue(cuts == expected_cuts)
        self.assertFalse(grain.cuts == expected_cuts)
        cuts = [Cut('s', [1, 1, 0], radius=3),
                Cut('s', [1, 2, 1], radius=4)]
        repeats = [1, 5, 2]
        grain = gc.Grain('test', unitcell, cuts, repeats)
        cuts = gc.alter_cuts(13, grain)
        cuts = [[cut.radius, cut.point.tolist()] for cut in cuts]
        expected_cuts = [[39, [13, 65, 0]], [52, [13, 130, 26]]]
        self.assertTrue(cuts == expected_cuts)
        self.assertFalse(grain.cuts == expected_cuts)

    def test_cut_grain(self):
        '''
        Does cut grain remove the expected atoms from the supercell within a
        grain definition object.
        '''
        test_basis = [
            Atom('Fe', 0.0, 0.0, 0.0), Atom('Fe', 0.5, 0.5, 0.0),
            Atom('Pt', 0.5, 0.0, 0.5), Atom('Pt', 0.0, 0.5, 0.5)]
        test_unitcell = UnitCell(
            test_basis, [3.82, 0, 0], [0, 3.82, 0], [0, 0, 3.711])
        test_supercell = SuperCell(test_unitcell, 10, 10, 10)
        test_cuts = [Cut('p', [5, 0, 0], plane=[1, 0, 0])]
        grain = gc.cut_grain(test_supercell, test_cuts)
        atom_array = grain.fractional
        self.assertTrue(len(atom_array) == 2200)
        maximums = np.max(atom_array['coordinates'], axis=0)
        self.assertTrue(maximums[0] <= 5)

    def test_best_composition(self):
        '''
        Test that the best composition found from a grain set is close to the
        ratio you would expect from the underlying supercell ratio.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        cuts_2 = [Cut('p', [1, 1, 0.5], plane=[1, 1, 1]),
                  Cut('p', [0, 1, 2], plane=[0, 0, 0.5])]
        repeat_ratio_2 = [1, 2, 1]
        grains = [gc.Grain('test', unitcell, cuts, repeat_ratio),
                  gc.Grain('test_2', unitcell, cuts_2, repeat_ratio_2)]
        grains = gc.scale_grains(grains, 5000)
        grains = gc.best_composition(grains, 5000)
        best_composition = grains[0].best_composition.values.tolist()[0]
        self.assertTrue(best_composition == [2503, 2497])

    def test_composition_columns(self):
        '''
        When given a particular grain, are the dtype list returned correctly?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        columns = gc.composition_columns(grain, float)
        self.assertTrue(columns == [('Fe', 'f8'), ('Pt', 'f8')])
        columns = gc.composition_columns(grain, int)
        self.assertTrue(columns == [('Fe', 'i8'), ('Pt', 'i8')])

    def test_remove_surface_atoms(self):
        '''
        Are the surface atoms removed down to the desired composition?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        composition = np.array([(550, 450)],
                               dtype=[('Fe', 'i8'), ('Pt', 'i8')])
        composition = pd.DataFrame(composition, index=[0])
        grain.best_composition = composition
        grain = gc.remove_surface_atoms(grain)
        composition = gc.get_composition(grain).values.tolist()
        self.assertTrue(composition[0] == [550, 450])

    def test_get_surface_atoms(self):
        '''
        Does the surface atom function correctly define the surface atoms for
        grains of arbitrary shape and composition?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5),
                      Atom('Zn', 0.1, 0.4, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        gc.get_surface_atoms(grain)
        percent_surface = (grain.surface_atoms.shape[0]/
                           grain.supercell.fractional.shape[0])
        self.assertTrue(np.isclose(percent_surface, 0.3, atol=0.05))
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        grain = gc.Grain('test2', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        gc.get_surface_atoms(grain)
        percent_surface = (grain.surface_atoms.shape[0]/
                           grain.supercell.fractional.shape[0])
        self.assertTrue(np.isclose(percent_surface, 0.3, atol=0.05))

    def test_get_composition(self):
        '''
        Is the correct composition of a given grain returned?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5),
                      Atom('Zn', 0.1, 0.4, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        composition = gc.get_composition(grain)
        self.assertTrue(composition.values.tolist() == [[600, 500, 500]])

    def test_remove_atoms(self):
        '''
        Are the correct atoms recorded for removal in the grain when various
        vertices are provided?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5),
                      Atom('Zn', 0.1, 0.4, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 10)
        surface_atoms = gc.get_surface_atoms(grain).surface_atoms
        vertices = np.array([400, 413])
        composition = np.array([(550, 450, 450)],
                               dtype=[('Fe', 'i8'), ('Pt', 'i8'),
                                      ('Zn', 'i8')])
        composition = pd.DataFrame(composition, index=[0])
        grain.composition_deltas = composition
        breaking = gc.remove_atoms(grain, surface_atoms, vertices)
        composition = grain.composition_deltas.values.tolist()[0]
        # Test few vertices many deletions.
        self.assertTrue(composition == [550, 450, 448])
        self.assertFalse(breaking)
        self.assertTrue(grain.surface_atoms[400]['removed'])
        self.assertTrue(grain.surface_atoms[413]['removed'])
        self.assertTrue(np.sum(grain.surface_atoms['removed']) == 2)
        # Test vertices of all types with many deletions
        vertices = np.array([33, 35, 320, 321, 405])
        breaking = gc.remove_atoms(grain, surface_atoms, vertices)
        composition = grain.composition_deltas.values.tolist()[0]

        self.assertTrue(composition == [548, 448, 447])
        self.assertFalse(breaking)
        self.assertTrue(grain.surface_atoms[33]['removed'])
        self.assertTrue(grain.surface_atoms[35]['removed'])
        self.assertTrue(grain.surface_atoms[320]['removed'])
        self.assertTrue(grain.surface_atoms[321]['removed'])
        self.assertTrue(grain.surface_atoms[405]['removed'])
        self.assertTrue(np.sum(grain.surface_atoms['removed']) == 7)
        # Test removal of vertices leading to correct composition
        vertices = np.array([401])
        composition = np.array([(0, 0, 1)],
                               dtype=[('Fe', 'i8'), ('Pt', 'i8'),
                                      ('Zn', 'i8')])
        composition = pd.DataFrame(composition, index=[0])
        grain.composition_deltas = composition
        breaking = gc.remove_atoms(grain, surface_atoms, vertices)
        composition = grain.composition_deltas.values.tolist()[0]
        self.assertTrue(composition == [0, 0, 0])
        self.assertTrue(breaking)
        self.assertTrue(grain.surface_atoms[401]['removed'])
        self.assertTrue(np.sum(grain.surface_atoms['removed']) == 8)
        # Test inability to remove any vertices leading to new surface
        vertices = np.array([403])
        composition = np.array([(1, 0, 0)],
                               dtype=[('Fe', 'i8'), ('Pt', 'i8'),
                                      ('Zn', 'i8')])
        composition = pd.DataFrame(composition, index=[0])
        grain.composition_deltas = composition
        breaking = gc.remove_atoms(grain, surface_atoms, vertices)
        composition = grain.composition_deltas.values.tolist()[0]
        self.assertTrue(composition == [1, 0, 0])
        self.assertTrue(breaking)
        self.assertTrue(np.sum(grain.surface_atoms['removed']) == 8)

    def test_update_supercell(self):
        '''
        Does the function correctly transfer a change in the surface atoms of
        the grain to its supercell?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5),
                      Atom('Zn', 0.1, 0.4, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], plane=[0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        gc.build_grain(grain, 10)
        gc.get_surface_atoms(grain)
        grain.surface_atoms[4]['removed'] = True
        index = grain.surface_atoms[4]['index']
        previous_atom = grain.supercell.fractional[index]
        gc.update_supercell(grain)
        self.assertFalse(grain.supercell.fractional[index] == previous_atom)




if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import grain_creation as gc
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    import testing_tools as test_tool
    import file_formatter as ff
    from edits import Cut
    unittest.main()
