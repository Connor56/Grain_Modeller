import unittest
import time
import os
import sys
from itertools import combinations
import numpy as np
import re


class TestFileFormatter(unittest.TestCase):

    def test_format_file(self):
        '''
        Does format_file correctly format and produce files for the LAMMPS
        data input file?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 1.1], [0, 0, 1])]
        repeat_ratio = [1, 1, 2]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grain.supercell = gc.build_grain(grain, 1)
        grain.border = 5
        template = 'LAMMPS_data_file'
        directory = '../grain_modeller/file_templates'
        template_after = ff.format_file(grain, template, directory)
        self.assertFalse(template == template_after)
        expected_string = ('test\n\n4 atoms\n2 atom types\n-5.0 6.5 xlo xhi\n'
                           +'-5.0 6.5 ylo yhi\n-5.0 9.5 zlo zhi\n0 0 0 xy xz '
                           +'yz\n\nAtoms\n\n1 1 0.0 0.0 0.0\n2 1 0.0 0.0 3.0\n'
                           +'3 2 1.5 1.5 1.5\n4 2 1.5 1.5 4.5\n\nMasses\n\n1 '
                           +'55.85\n2 195.08\n\n')
        self.assertTrue(template_after == expected_string)

    def test_read_file(self):
        '''
        Does the read file function correctly take in a file type and return
        the file string to be formatted, along with the data required to
        format? Does it return an error when the type can't be found?
        '''
        template = 'LAMMPS_data_file'
        directory = '../grain_modeller/file_templates'
        template, variables = ff.read_file(template, directory)
        template_exists = list(re.finditer('TEMPLATE', template))
        self.assertTrue(template_exists == [])
        for key in variables['Required']:
            key_exists = list(re.finditer(key, template))
            self.assertFalse(key_exists == [])
            key_exists = list(re.finditer('--#'+key+'#--', template))
            self.assertFalse(key_exists == [])
        self.assertTrue(template.split('\n')[0] == '--#Name#--')
        self.assertTrue(template.split('\n')[-2] == 'POSSIBLE:Potentials')
        self.assertTrue(template.split('\n')[-1] == '')
        for key in variables['Possible']:
            key_exists = list(re.finditer(key, template))
            self.assertFalse(key_exists == [])
            key_exists = list(re.finditer('POSSIBLE:'+key, template))
            self.assertFalse(key_exists == [])

    def test_check_formatting_object(self):
        '''
        Does check formatting object correctly direct the format checks based
        on type, and does it stop objects that lack the correct variables from
        being let through?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], [0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grains = gc.compositionally_match(grain, 10000)
        variables = {'Required': {
            'Name': None, 'Number_of_Atoms': None,
            'Number_of_Atom_Types': None, 'X_Box_Minimum': None,
            'X_Box_Maximum': None, 'Y_Box_Minimum': None,
            'Y_Box_Maximum': None, 'Z_Box_Minimum': None,
            'Z_Box_Maximum': None, 'XY': None, 'XZ': None, 'YZ': None,
            'Atoms_Cartesian_LAMMPS': None, 'Masses_LAMMPS': None},
            'Possible': {'Potentials': None}}
        formatting_object = ff.check_formatting_object(grains[0], variables)

    def test_format_grain(self):
        '''
        Does format grain correctly get the atoms list in the right format?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        cuts = [Cut('p', [0, 0, 0.5], [0, 0, 1])]
        repeat_ratio = [1, 1, 1]
        grain = gc.Grain('test', unitcell, cuts, repeat_ratio)
        grains = gc.compositionally_match(grain, 400)
        formatting_object = ff.format_grain(grains[0])
        keys = list(formatting_object.keys())
        expected_keys = [
            'Name', 'Number_of_Atoms', 'Number_of_Atom_Types', 'X_Box_Minimum',
            'X_Box_Maximum', 'Y_Box_Minimum', 'Y_Box_Maximum', 'Z_Box_Minimum',
            'Z_Box_Maximum', 'XY', 'XZ', 'YZ', 'Atoms_Fractional',
            'Atoms_Cartesian', 'Atoms_Cartesian_LAMMPS', 'Masses',
            'Masses_LAMMPS']
        self.assertTrue(keys == expected_keys)
        self.assertTrue(formatting_object['Number_of_Atoms'] == 400)

    def test_array_to_string(self):
        '''
        Does array to string correctly transfer a structured atom array to a
        string which can be used as simulation input?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 2, 1, 2)
        string = ff.array_to_string(supercell.fractional)
        expected_string = ('Fe 0.0 0.0 0.0\nFe 0.0 0.0 1.0\nFe 1.0 0.0 0.0\n'
                           + 'Fe 1.0 0.0 1.0\nPt 0.5 0.5 0.5\nPt 0.5 0.5 1.5\n'
                           + 'Pt 1.5 0.5 0.5\nPt 1.5 0.5 1.5')
        self.assertTrue(string == expected_string)
        self.assertFalse(type(supercell.fractional) == type(string))

    def test_speed_of_array_to_string(self):
        '''
        Test array to string for a very large supercell. Check it takes less
        than 2 seconds to create a string of 2million atoms.
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 100, 100, 100)
        supercell.set_cartesian()
        begin = time.time()
        ff.array_to_string(supercell.cartesian)
        self.assertTrue(time.time()-begin < 8)

    def test_format_string(self):
        '''
        Does a formatting object correctly format a template string?
        '''
        template = ('--#BOOHOO#-- You Died\nYou did score: --#Score#-- though.'
                    + '\nPOSSIBLE:Enemy\n\nIn fairness I guess it isnt too '
                    + 'bad. You could be like --#Enemy#--.\nPOSSIBLE:Enemy')
        formatting_object = {
            'Required': {'BOOHOO': 'Aww Shucks', 'Score': 456},
            'Possible': {'Enemy': None}}
        template_after = ff.format_string(template, formatting_object)
        expected_string = 'Aww Shucks You Died\nYou did score: 456 though.'
        self.assertTrue(template_after == expected_string)
        self.assertFalse(template == template_after)
        formatting_object = {
            'Required': {'BOOHOO': 'Aww Shucks', 'Score': 456},
            'Possible': {'Enemy': 'Gareth'}}
        template_after = ff.format_string(template, formatting_object)
        expected_string = ('Aww Shucks You Died\nYou did score: 456 though.'
                           + '\n\nIn fairness I guess it isnt too bad. You '
                           + 'could be like Gareth.')
        self.assertFalse(template == template_after)
        self.assertTrue(template_after == expected_string)

    def test_format_string_works_with_nested_variables(self):
        '''
        Check nested variables can be used so that one not being used doesn't
        wipe out all the rest.
        '''
        self.assertTrue(False)

    def test_format_supercell(self):
        '''
        Does a supercell get correctly formatted?
        '''
        test_basis = [Atom('Fe', 0, 0, 0), Atom('Pt', 0.5, 0.5, 0.5)]
        unitcell = UnitCell(test_basis, [3, 0, 0], [0, 3, 0], [0, 0, 3])
        supercell = SuperCell(unitcell, 10, 10, 10)
        formatting_object = ff.format_supercell(supercell)
        keys = list(formatting_object.keys())
        expected_keys = [
            'Name', 'Number_of_Atoms', 'Number_of_Atom_Types', 'X_Box_Minimum',
            'X_Box_Maximum', 'Y_Box_Minimum', 'Y_Box_Maximum', 'Z_Box_Minimum',
            'Z_Box_Maximum', 'XY', 'XZ', 'YZ', 'Atoms_Fractional',
            'Atoms_Cartesian', 'Atoms_Cartesian_LAMMPS', 'Masses',
            'Masses_LAMMPS']
        self.assertTrue(keys == expected_keys)
        self.assertTrue(formatting_object['Number_of_Atoms'] == 2000)
        self.assertTrue(formatting_object['Y_Box_Maximum'] == 28.5)













if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import file_formatter as ff
    import grain_creation as gc
    from atom import Atom
    from unitcell import UnitCell
    from supercell import SuperCell
    from edits import Cut
    unittest.main()
