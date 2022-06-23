import unittest
import os
import sys

class TestInterface(unittest.TestCase):

    def test_match_unitcells_correctly_matches_unitcell_interfaces(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,4,0],[0,0,4])
        test_unitcell2 = test_unitcell.copy()
        sim = interface.Interface()
        expected_output = {'Cell_1_X_Repeat': 1, 'Cell_1_Y_Repeat': 1,
                           'Cell_2_X_Repeat': 1, 'Cell_2_Y_Repeat': 1}
        replications = sim.match_cells(test_unitcell, test_unitcell2, 200)
        self.assertTrue(replications == expected_output)

    def test_match_unitcells_correctly_matches_unitcell_interfaces_x(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis,[4,0,0],[0,13,0],
                                            [0,0,22])
        test_unitcell2 = ge.UnitCellExtended(test_basis,[4,0,0],[0,4,0],
                                             [0,0,4])
        sim = interface.Interface()
        expected_output = {'Cell_1_X_Repeat': 1, 'Cell_1_Y_Repeat': 4,
                           'Cell_2_X_Repeat': 1, 'Cell_2_Y_Repeat': 13}
        replications = sim.match_cells(test_unitcell, test_unitcell2, 200)
        self.assertTrue(replications == expected_output)

    def test_match_unitcells_correctly_matches_unitcell_interfacesx_and_y(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis,[17,0,0],[0,13,0],
                                            [0,0,1])
        test_unitcell2 = ge.UnitCellExtended(test_basis,[6,0,0],[0,12,0],
                                             [0,0,1])
        sim = interface.Interface()
        expected_output = {'Cell_1_X_Repeat': 6, 'Cell_1_Y_Repeat': 12,
                           'Cell_2_X_Repeat': 17, 'Cell_2_Y_Repeat': 13}
        replications = sim.match_cells(test_unitcell, test_unitcell2, 200)
        self.assertTrue(replications == expected_output)

    def test_match_unitcells_correctly_matches_unitcell_interfaces_similar(
            self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis,[5,0,0],[0,6.09,0],
                                            [0,0,1])
        test_unitcell2 = ge.UnitCellExtended(test_basis, [5.04,0,0],[0,6,0],
                                             [0,0,1])
        sim = interface.Interface()
        expected_output = {'Cell_1_X_Repeat': 1, 'Cell_1_Y_Repeat': 1,
                           'Cell_2_X_Repeat': 1, 'Cell_2_Y_Repeat': 1}
        replications = sim.match_cells(test_unitcell, test_unitcell2, 200)
        self.assertTrue(replications == expected_output)

    def test_match_cells_only_takes_in_two_unit_cells(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,4,0],[0,0,4])
        test_unitcell2 = [1,2]
        sim = interface.Interface()
        self.assertRaisesRegex(TypeError,
                               'Unitcells must be of type: UnitCellExtended or'
                               + ' UnitCellExtendedFile.', sim.match_cells,
                               test_unitcell, test_unitcell2, 200)

    def test_get_correct_unitcell_gives_accurate_unitcell_100(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,6,0],[0,0,7])
        sim = interface.Interface()
        simulation_type = 'Energy_Minimisation'
        potential_types = [['Morse', 'Fe', 'Fe']]
        unitcell_list = {'Cell': test_unitcell, 'Potentials': potential_types,
                    'Simulation': simulation_type, 'Normal': [1,0,0]}
        new_unitcell = sim.get_correct_unitcell(unitcell_list)
        expected_attributes = {'a_Lattice_Parameter': 2.415223,
                               'b_Lattice_Parameter': 2.415231,
                               'c_Lattice_Parameter': 2.415209,
                               'Alpha': 90.0, 'Beta': 90.0, 'Gamma': 90.0,
                               'a_Lattice_Vector': [2.415223, 0.0, 0.0],
                               'b_Lattice_Vector': [0.0, 2.415231, 0.0],
                               'c_Lattice_Vector': [0.0, 0.0, 2.415209],
                               'UnitCell_X_Position': 0,
                               'UnitCell_Y_Position': 0,
                               'UnitCell_Z_Position': 0,
                               'Atom_1': ['Fe', 0.0, 0.0, 0.0]}
        self.assertTrue(new_unitcell.get_attributes() == expected_attributes)

    def test_get_correct_unitcell_gives_accurate_unitcell_001(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,6,0],[0,0,7])
        sim = interface.Interface()
        simulation_type = 'Energy_Minimisation'
        potential_types = [['Morse', 'Fe', 'Fe']]
        unitcell_list = {'Cell': test_unitcell, 'Potentials': potential_types,
                    'Simulation': simulation_type, 'Normal': [0,0,1]}
        new_unitcell = sim.get_correct_unitcell(unitcell_list)
        expected_attributes = {'a_Lattice_Parameter': 2.415231,
                               'b_Lattice_Parameter': 2.415209,
                               'c_Lattice_Parameter': 2.415223,
                               'Alpha': 90.0, 'Beta': 90.0, 'Gamma': 90.0,
                               'a_Lattice_Vector': [2.415231, 0.0, 0.0],
                               'b_Lattice_Vector': [0.0, 2.415209, 0.0],
                               'c_Lattice_Vector': [0.0, 0.0, 2.415223],
                               'UnitCell_X_Position': 0,
                               'UnitCell_Y_Position': 0,
                               'UnitCell_Z_Position': 0,
                               'Atom_1': ['Fe', 0.0, 0.0, 0.0]}
        self.assertTrue(new_unitcell.get_attributes() == expected_attributes)

    def test_get_correct_unitcell_gives_accurate_unitcell_010(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,6,0],[0,0,7])
        sim = interface.Interface()
        simulation_type = 'Energy_Minimisation'
        potential_types = [['Morse', 'Fe', 'Fe']]
        unitcell_list = {'Cell': test_unitcell, 'Potentials': potential_types,
                    'Simulation': simulation_type, 'Normal': [0,1,0]}
        new_unitcell = sim.get_correct_unitcell(unitcell_list)
        expected_attributes = {'a_Lattice_Parameter': 2.415223,
                               'b_Lattice_Parameter': 2.415209,
                               'c_Lattice_Parameter': 2.415231,
                               'Alpha': 90.0, 'Beta': 90.0, 'Gamma': 90.0,
                               'a_Lattice_Vector': [2.415223, 0.0, 0.0],
                               'b_Lattice_Vector': [0.0, 2.415209, 0.0],
                               'c_Lattice_Vector': [0.0, 0.0, 2.415231],
                               'UnitCell_X_Position': 0,
                               'UnitCell_Y_Position': 0,
                               'UnitCell_Z_Position': 0,
                               'Atom_1': ['Fe', 0.0, 0.0, 0.0]}
        self.assertTrue(new_unitcell.get_attributes() == expected_attributes)

    def test_create_interface_correctly_creates_an_interface(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [4,0,0],[0,4,0],[0,0,4])
        test_unitcell2 = test_unitcell.copy()
        sim = interface.Interface()
        repeat = {'Cell_1_X_Repeat': 1, 'Cell_1_Y_Repeat': 1,
                  'Cell_2_X_Repeat': 1, 'Cell_2_Y_Repeat': 1}
        atom_list = sim.create_interface_atom_list(test_unitcell,
                                                   test_unitcell2,repeat,8)
        expected_list = [['Fe', 0.0, 0.0, 0.0], ['Fe', 0.0, 0.0, 4.0],
                         ['Fe', 0.0, 0.0, 4.0], ['Fe', 0.0, 0.0, 8.0]]
        self.assertTrue(atom_list['Full_List'] == expected_list)

    def test_create_interface_correctly_creates_interface_large_unitcell(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Nd',0.0,0.0,0.0),ge.Atom('Nd',0.5,0.5,0.5),
            ge.Atom('Fe',0.356,0.0,0.0),ge.Atom('Fe',0.8568,0.5,0.5),
            ge.Atom('Fe',0.6432,0.0,0.0),ge.Atom('Fe',0.1432,0.5,0.5),
            ge.Atom('Fe',0.0,0.6432,0.0),ge.Atom('Fe',0.5,0.1432,0.5),
            ge.Atom('Fe',0.0,0.3568,0.0),ge.Atom('Fe',0.5,0.8568,0.5),
            ge.Atom('Fe',0.2728,0.5,0.0),ge.Atom('Fe',0.7728,0.0,0.5),
            ge.Atom('Fe',0.2272,0.0,0.5),ge.Atom('Fe',0.7272,0.5,0.0),
            ge.Atom('Fe',0.0,0.2272,0.5),ge.Atom('Fe',0.5,0.7272,0.0),
            ge.Atom('Fe',0.5,0.2728,0.0),ge.Atom('Fe',0.0,0.7728,0.5),
            ge.Atom('Fe',0.25,0.25,0.25),ge.Atom('Fe',0.75,0.75,0.75),
            ge.Atom('Fe',0.25,0.25,0.75),ge.Atom('Fe',0.75,0.75,0.25),
            ge.Atom('Fe',0.75,0.25,0.25),ge.Atom('Fe',0.25,0.75,0.75),
            ge.Atom('Fe',0.25,0.75,0.25),ge.Atom('Fe',0.75,0.25,0.75)]
        test_unitcell = ge.UnitCellExtended(test_basis,[8.611,0,0],[0,8.611,0],
                                            [0,0,4.802])
        test_unitcell2 = test_unitcell.copy()
        repeat = {'Cell_1_X_Repeat': 5, 'Cell_1_Y_Repeat': 6,
                  'Cell_2_X_Repeat': 6, 'Cell_2_Y_Repeat': 4}
        atom_list = sim.create_interface_atom_list(test_unitcell,test_unitcell2,
                                                   repeat, 4.802)
        sim.create_xyz_file('atom_list_file', atom_list['Full_List'])
        with open('test_files/atom_list_file_check.xyz', 'r') as file:
            expected_output = file.read()[:-1]
        with open('atom_list_file.xyz', 'r') as file:
            output = file.read()
        self.assertTrue(output == expected_output)
        os.system('rm atom_list_file.xyz')

    def test_complicated_interface_combination_you_get_correct_interface(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Nd',0.0,0.0,0.0),ge.Atom('Nd',0.5,0.5,0.5),
            ge.Atom('Fe',0.356,0.0,0.0),ge.Atom('Fe',0.8568,0.5,0.5),
            ge.Atom('Fe',0.6432,0.0,0.0),ge.Atom('Fe',0.1432,0.5,0.5),
            ge.Atom('Fe',0.0,0.6432,0.0),ge.Atom('Fe',0.5,0.1432,0.5),
            ge.Atom('Fe',0.0,0.3568,0.0),ge.Atom('Fe',0.5,0.8568,0.5),
            ge.Atom('Fe',0.2728,0.5,0.0),ge.Atom('Fe',0.7728,0.0,0.5),
            ge.Atom('Fe',0.2272,0.0,0.5),ge.Atom('Fe',0.7272,0.5,0.0),
            ge.Atom('Fe',0.0,0.2272,0.5),ge.Atom('Fe',0.5,0.7272,0.0),
            ge.Atom('Fe',0.5,0.2728,0.0),ge.Atom('Fe',0.0,0.7728,0.5),
            ge.Atom('Fe',0.25,0.25,0.25),ge.Atom('Fe',0.75,0.75,0.75),
            ge.Atom('Fe',0.25,0.25,0.75),ge.Atom('Fe',0.75,0.75,0.25),
            ge.Atom('Fe',0.75,0.25,0.25),ge.Atom('Fe',0.25,0.75,0.75),
            ge.Atom('Fe',0.25,0.75,0.25),ge.Atom('Fe',0.75,0.25,0.75)]
        test_unitcell = ge.UnitCellExtended(test_basis,[8.611,0,0],[0,8.611,0],
                                            [0,0,4.802])
        test_supercell = ge.SuperCellExtended(test_unitcell, 6, 6, 6)
        potential_types = [['Morse', 'Fe', 'Fe'], ['Morse', 'Nd', 'Fe'],
                           ['Morse', 'Nd', 'Nd']]
        simulation_type = 'Energy_Minimisation'
        test_unitcell2 = test_supercell.define_new_cubic_unitcell([1,1,1],
                         [2,2,2], potential_types, simulation_type)
        repeat = sim.match_cells(test_unitcell, test_unitcell2, 200)
        atom_list = sim.create_interface_atom_list(test_unitcell,
            test_unitcell2, repeat, 20)
        sim.create_xyz_file('complicated_interface_test',
                            atom_list['Full_List'])
        with open('complicated_interface_test.xyz', 'r') as file:
            output = file.read()
        with open('test_files/complicated_interface_test_check.xyz', 'r') \
                as file:
            expected_output = file.read()[:-1]
        self.assertTrue(output == expected_output)
        os.system('rm complicated_interface_test.xyz')
        sim.create_lammps_data_file('interface_test', potential_types,
                                    atom_list['Full_List'])

    def test_get_z_repeat_for_unitcell(self):
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(
            test_basis, [5,0,0], [0,5,0], [0,0,5])
        sim = interface.Interface()
        z_repeat = sim.get_z_repeat_for_unitcell(test_unitcell, 50)
        self.assertTrue(z_repeat == 10)
        z_repeat = sim.get_z_repeat_for_unitcell(test_unitcell, 36)
        self.assertTrue(z_repeat == 7)

    def test_get_interface_bounds_correctly_gets_the_expected_bounds(self):
        sim = interface.Interface()
        supercell_attributes_1 = {'X_Repeat': 6, 'Y_Repeat': 6, 'Z_Repeat': 6,
                'a_Lattice_Vector': [51.666000000000004, 0.0, 0.0],
                'b_Lattice_Vector': [0.0, 51.666000000000004, 0.0],
                'c_Lattice_Vector': [0.0, 0.0, 28.811999999999998]}
        supercell_attributes_2 = {'X_Repeat': 6, 'Y_Repeat': 6, 'Z_Repeat': 6,
                'a_Lattice_Vector': [51.666000000000004, 0.0, 0.0],
                'b_Lattice_Vector': [4.5, 51.666000000000004, 0.0],
                'c_Lattice_Vector': [-6.0, -3.0, 28.811999999999998]}
        interface_bounds = sim.get_interface_bounds(supercell_attributes_1,
                           supercell_attributes_2)
        expected_interface_bounds = {'X_Min': 4.4999, 'X_Max':
                                    45.66610000000001, 'Y_Min': -0.0001,
                                    'Y_Max': 48.66610000000001}
        self.assertTrue(interface_bounds == expected_interface_bounds)

    def test_get_interface_bounds_correctly_gets_the_expected_bounds2(self):
        sim = interface.Interface()
        supercell_attributes_1 = {'X_Repeat': 6, 'Y_Repeat': 6, 'Z_Repeat': 6,
                'a_Lattice_Vector': [40, 0.0, 0.0],
                'b_Lattice_Vector': [-10.0, 32, 0.0],
                'c_Lattice_Vector': [-10.0, 10.0, 13]}
        supercell_attributes_2 = {'X_Repeat': 6, 'Y_Repeat': 6, 'Z_Repeat': 6,
                'a_Lattice_Vector': [51.666000000000004, 0.0, 0.0],
                'b_Lattice_Vector': [4.5, 51.666000000000004, 0.0],
                'c_Lattice_Vector': [-6.0, -3.0, 28.811999999999998]}
        interface_bounds = sim.get_interface_bounds(supercell_attributes_1,
                           supercell_attributes_2)
        expected_interface_bounds = {'X_Min': 4.4999, 'X_Max':
                                    20.0001, 'Y_Min': 9.9999,
                                    'Y_Max': 32.0001}
        self.assertTrue(interface_bounds == expected_interface_bounds)

    def test_get_box_bounds_interface_correctly_gets_box_boundaries(self):
        test_atom_list = [['Nd', 1,2,3]]
        sim = interface.Interface()
        box_bounds = sim.get_box_bounds_interface(test_atom_list)
        expected_output = {'X_Min': 0.9, 'X_Max': 1.1, 'Y_Min': 1.9,
                           'Y_Max': 2.1, 'Z_Min': 3, 'Z_Max': 3,
                           'XY_Length': '0', 'XZ_Length': '0', 'YZ_Length': '0'}
        self.assertTrue(expected_output == box_bounds)

    def test_get_box_bounds_interface_correctly_gets_box_boundaries2(self):
        test_atom_list = [['Nd', 1,2,3],['Nd', -2.3,2,2],['Nd', 4839,-2,31],
                          ['Nd', 1,1,3]]
        sim = interface.Interface()
        box_bounds = sim.get_box_bounds_interface(test_atom_list)
        expected_output = {'X_Min': -2.4, 'X_Max': 4839.1, 'Y_Min': -2.1,
                           'Y_Max': 2.1, 'Z_Min': 2, 'Z_Max': 31,
                           'XY_Length': '0', 'XZ_Length': '0', 'YZ_Length': '0'}
        self.assertTrue(expected_output == box_bounds)

    def test_create_orientation_input(self):
        sim = interface.Interface()
        potential_types = [
            ['Morse','Fe','Fe'],['Morse','Nd','Fe'],['Morse','Nd','Nd']]
        box_bounds = {
            'X_Min': 2.2399999999999998, 'X_Max': 34.1,
            'Y_Min': 1.0999999999999999, 'Y_Max': 34.1, 'Z_Min': 0.348,
            'Z_Max': 3, 'XY_Length': '0', 'XZ_Length': '0', 'YZ_Length': '0'}
        cut_off = 12
        cell_sizes = {'Max_X_Unitcell_Length': 2, 'Max_Y_Unitcell_Length': 2}
        types = ['1','2']
        sim.create_orientation_input(
            potential_types, box_bounds, cut_off, cell_sizes, types)
        with open('orientate_boundary.min', 'r') as file:
            orientate_input = file.read()
        with open('test_files/orientate_boundary_check.min', 'r') as file:
            expected_orientate_input = file.read()[:-1]
        self.assertTrue(orientate_input == expected_orientate_input)
        os.system('rm orientate_boundary.min')

    def OFF_test_create_interface_input_correctly_creates_lammps_file(self):
        sim = interface.Interface()
        potential_types = [['Morse', 'Fe', 'Fe'], ['Morse', 'Fe', 'Nd'],
                           ['Morse', 'Nd', 'Nd']]
        box_bounds = {'X_Min': '-3.608224830031759e-16',
                      'X_Max': '77.33873310999999',
                      'Y_Min': '-3.608224830031759e-16',
                      'Y_Max': '172.50527526399998',
                      'Z_Min': '0.0', 'Z_Max': '42.095388', 'XY_Length': '0',
                      'XZ_Length': '0', 'YZ_Length': '0'}
        os.system('mkdir test_interface')
        types = ['1', '2']
        sim.create_interface_input(box_bounds, 'test_interface',
                                   potential_types, types)
        with open('test_files/test_interface_check/test_interface_check'+
                '.interface', 'r') as file:
            expected_input = file.read()[:-1].split('\n')
        with open('test_interface/test_interface.interface', 'r') as file:
            input = file.read().split('\n')
        self.assertTrue(input == expected_input)
        os.system('rm -r test_interface')

    def test_get_groups_and_computes_returns_correct_output(self):
        sim = interface.Interface()
        types = ['1']
        group_list = ['interface']
        groups_and_computes = sim.get_groups_and_computes_input(
                              types, group_list, 'check')
        expected_groups_and_computes = ['group atom_type_1 type 1',
            '\n#Atom interface Type 1 Computes',
            'group type_1_interface_atoms intersect atom_type_1 '
            + 'check_interface_atoms',
            'compute type_1_interface_energy type_1_interface_atoms reduce '
            + 'sum c_per_atom',
            'variable number_of_type_1_interface_atoms equal '
            + 'count(type_1_interface_atoms)',
            'variable interface_type_1_energy_per_atom equal '
            + 'c_type_1_interface_energy/v_number_of_type_1_interface_atoms']
        self.assertTrue(groups_and_computes == expected_groups_and_computes)

    def test_get_groups_and_computes_returns_correct_output_complicated(self):
        sim = interface.Interface()
        types = ['1','2','3']
        group_list = ['interface_1', 'interface_2']
        groups_and_computes = sim.get_groups_and_computes_input(
                              types, group_list, 'unfrozen')
        groups_and_computes = '\n'.join(groups_and_computes)
        groups_and_computes = groups_and_computes.split('\n')
        with open('test_files/groups_and_computes_input_check.in', 'r') as file:
            expected_groups_and_computes = file.read()[:-1].split('\n')
        self.assertTrue(groups_and_computes == expected_groups_and_computes)

    def test_stack_supercells(self):
        test_basis = [ge.Atom('Nd',0.0,0.0,0.0),ge.Atom('Nd',0.5,0.5,0.5),
            ge.Atom('Fe',0.356,0.0,0.0),ge.Atom('Fe',0.8568,0.5,0.5),
            ge.Atom('Fe',0.6432,0.0,0.0),ge.Atom('Fe',0.1432,0.5,0.5)]
        test_unitcell = ge.UnitCellExtended(test_basis,[8.611,0,0],[0,8.611,0],
                                            [0,0,4])
        test_supercell = ge.SuperCellExtended(test_unitcell, 1, 1, 1)
        test_basis_2 = [ge.Atom('Nd',0.0,0.0,0.0),ge.Atom('Nd',0.5,0.5,0.5),
            ge.Atom('Fe',0.356,0.0,0.0),ge.Atom('Fe',0.8568,0.5,0.5),
            ge.Atom('Fe',0.6432,0.0,0.0),ge.Atom('Fe',0.1432,0.5,0.5)]
        test_unitcell_2 = ge.UnitCellExtended(test_basis,[4,0,0],[0,4,0],
                                              [0,0,4.802])
        number_of_atoms_created = ge.Atom.number_of_atoms_created+1
        test_supercell_2 = ge.SuperCellExtended(test_unitcell_2, 1,1,1)
        sim = interface.Interface()
        atom_lists = sim.stack_supercells(test_supercell, test_supercell_2)
        expected_cell_2 = [
            ['Atom'+str(number_of_atoms_created), 'Nd', 0.0, 0.0, 2.0],
            ['Atom'+str(number_of_atoms_created+1), 'Nd', 2.0, 2.0, 4.401],
            ['Atom'+str(number_of_atoms_created+2), 'Fe', 1.424, 0.0, 2.0],
            ['Atom'+str(number_of_atoms_created+3), 'Fe', 3.4272, 2.0, 4.401],
            ['Atom'+str(number_of_atoms_created+4), 'Fe', 2.5728, 0.0, 2.0],
            ['Atom'+str(number_of_atoms_created+5), 'Fe', 0.5728, 2.0, 4.401]]
        self.assertTrue(atom_lists['SuperCell_2'] == expected_cell_2)

    def test_trim_supercell_atoms_works_correctly(self):
        supercell_1_atom_list = [['Nd', 2.0, 2.0, 22.401],
                                 ['Fe', 1.424, 0.0, 20.0],
                                 ['Fe', 3.4272, 2.0, 22.401]]
        supercell_2_atom_list = [['Fe', 2.5728, 0.0, 20.0],
                                 ['Fe', 0.5728, 2.0, 22.401]]
        box_bounds = {'X_Min': 1.9,'X_Max': 3,'Y_Min': -0.1,'Y_Max':2.1}
        sim = interface.Interface()
        atom_lists = sim.trim_supercell_atoms(
            supercell_1_atom_list, supercell_2_atom_list, box_bounds)
        expected_atom_lists = {
            'SuperCell_1': [['Nd', 2.0, 2.0, 22.401]],
            'SuperCell_2': [['Fe', 2.5728, 0.0, 20.0]],
            'Full_List': [['Nd', 2.0, 2.0, 22.401], ['Fe', 2.5728, 0.0, 20.0]]}
        self.assertTrue(atom_lists == expected_atom_lists)

    def test_format_supercell_atom_list(self):
        atom_lists = {'Full_List':
                     [['Atom48003', 'Nd', 0.0, 0.0, 20.0],
                      ['Atom48004', 'Nd', 2.0, 2.0, 22.401],
                      ['Atom48005', 'Fe', 1.424, 0.0, 20.0],
                      ['Atom48006', 'Fe', 3.4272, 2.0, 22.401],
                      ['Atom48007', 'Fe', 2.5728, 0.0, 20.0],
                      ['Atom48008', 'Fe', 0.5728, 2.0, 22.401]],
                      'SuperCell_1': [['Atom294893', 'Fe', 1, 2, 3]],
                      'SuperCell_2': [['Atom2946593', 'Nd', 2.34, 1.2, .098]]}
        sim = interface.Interface()
        atom_lists = sim.format_supercell_atom_lists(atom_lists)
        expected_output = {'Full_List':
                     [['Fe', 0.5728, 2.0, 22.401],
                      ['Fe', 1.424, 0.0, 20.0],
                      ['Fe', 2.5728, 0.0, 20.0],
                      ['Fe', 3.4272, 2.0, 22.401],
                      ['Nd', 0.0, 0.0, 20.0],
                      ['Nd', 2.0, 2.0, 22.401]],
                      'SuperCell_1': [['Fe', 1, 2, 3]],
                      'SuperCell_2': [['Nd', 2.34, 1.2, .098]]}
        self.assertTrue(expected_output == atom_lists)

    def test_match_repeats_to_max_side_length_gives_accurate_cell_repeats(self):
        test_basis = [ge.Atom('Nd',0.0,0.0,0.0),ge.Atom('Nd',0.5,0.5,0.5),
            ge.Atom('Fe',0.356,0.0,0.0),ge.Atom('Fe',0.8568,0.5,0.5),
            ge.Atom('Fe',0.6432,0.0,0.0),ge.Atom('Fe',0.1432,0.5,0.5)]
        test_unitcell = ge.UnitCellExtended(test_basis,[8.611,0,0],[0,8.611,0],
                                            [0,0,4])
        max_side_length = 3400
        repeat = {'Cell_1_X_Repeat': 12, 'Cell_1_Y_Repeat': 3,
                  'Cell_2_X_Repeat': 43, 'Cell_2_Y_Repeat': 23}
        sim = interface.Interface()
        repeat = sim.match_repeats_to_max_side_length(test_unitcell, repeat,
                                                      max_side_length)
        expected_repeat = {'Cell_1_X_Repeat': 384, 'Cell_1_Y_Repeat': 393,
                           'Cell_2_X_Repeat': 1376, 'Cell_2_Y_Repeat': 3013}
        self.assertTrue(repeat == expected_repeat)

    def test_shift_supercell_works_correctly(self):
        atom_list = [['Fe', 0.5728, 2.0, 22.401],
                     ['Fe', 1.424, 0.0, 20.0],
                     ['Fe', 2.5728, 0.0, 20.0],
                     ['Fe', 3.4272, 2.0, 22.401],
                     ['Nd', 0.0, 0.0, 20.0],
                     ['Nd', 2.0, 2.0, 22.401]]
        shifts = {'X_Shift': 23, 'Y_Shift': 1, 'Z_Shift': 2}
        sim = interface.Interface()
        atom_list = sim.shift_supercell(atom_list, shifts)
        expected_atom_list = [['Fe', 23.5728, 3.0, 24.401],
                              ['Fe', 24.424, 1.0, 22.0],
                              ['Fe', 25.5728, 1.0, 22.0],
                              ['Fe', 26.4272, 3.0, 24.401],
                              ['Nd', 23.0, 1.0, 22.0],
                              ['Nd', 25.0, 3.0, 24.401]]
        self.assertTrue(atom_list == expected_atom_list)

    def test_realign_interface_works_correctly(self):
        atom_lists = {'Full_List':
                     [['Nd', 0.0, 0.0, 20.0],
                      ['Nd', 2.0, 2.0, 22.401],
                      ['Fe', 1.424, 0.0, 20.0],
                      ['Fe', 3.4272, 2.0, 22.401],
                      ['Fe', 2.5728, 0.0, 20.0],
                      ['Fe', 0.5728, 2.0, 22.401]],
                      'SuperCell_1': [['Fe', 1, 2, 3]],
                      'SuperCell_2': [['Nd', 2.34, 1.2, .098]]}
        box_bounds = {'X_Min': 1, 'X_Max': 2.2, 'Y_Min': 23, 'Y_Max': 1}
        sim = interface.Interface()
        atom_lists = sim.realign_interface(atom_lists, box_bounds)
        expected_atom_lists = {
            'Full_List': [['Fe', 1-1, 2-23, 3],['Nd', 2.34-1, 1.2-23, .098]],
            'SuperCell_1': [['Fe', 1-1, 2-23, 3]],
            'SuperCell_2': [['Nd', 2.34-1, 1.2-23, .098]]}
        self.assertTrue(atom_lists == expected_atom_lists)

    def test_create_standard_lammps_output_creates_correct_files(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [3,0,0], [1,2,0],
                                            [-2,1,2])
        test_unitcell_2 = test_unitcell.copy()
        potential_types = [['Morse', 'Fe', 'Fe']]
        os.system('mkdir test_interface')
        types = ['1']
        sim.create_standard_lammps_input('test_interface', test_unitcell,
            test_unitcell_2, potential_types, types)
        with open('test_files/cell_1_only_check.in', 'r') \
                as file:
            cell_1_expected_input = file.read()[:-1]
        with open('test_files/cell_2_only_check.in', 'r') \
                as file:
            cell_2_expected_input = file.read()[:-1]
        with open('test_interface/cell_1_only.in', 'r') as \
                file:
            cell_1_input = file.read()
        with open('test_interface/cell_2_only.in', 'r') as \
                file:
            cell_2_input = file.read()
        self.assertTrue(cell_1_input == cell_1_expected_input)
        self.assertTrue(cell_2_input == cell_2_expected_input)
        os.system('rm -r test_interface')

    def test_add_single_file_only_gives_correct_input_command(self):
        sim = interface.Interface()
        standard_input = ['read_data']
        standard_input = sim.add_single_file_to_read(standard_input, 1)
        expected_standard_input = ['read_data cell_1_only.in']
        self.assertTrue(standard_input == expected_standard_input)

    def OFF_test_create_input_for_single_LAMMPS_simulation(self):
        sim = interface.Interface()
        potential_types = [['Morse', 'Fe', 'Fe']]
        unitcell_number = 1
        types = ['1']
        standard_input = sim.create_input_for_single_LAMMPS_simulation(
            'cell_1_only', unitcell_number, potential_types, types)
        with open('test_files/standard_input_for_single_lammps_file.check') \
                as file:
            expected_standard_input = file.read()[:-1].split('\n')
        standard_input = '\n'.join(standard_input)
        standard_input = standard_input.split('\n')
        self.assertTrue(standard_input == expected_standard_input)

    def test_set_potential_style_gives_correct_potential(self):
        sim = interface.Interface()
        standard_input = ['pair_style']
        potential_types = [['Morse', 'Fe','Fe']]
        standard_input = sim.set_potential_style(
            standard_input, potential_types)
        expected_standard_input = ['pair_style morse 12.0']
        self.assertTrue(standard_input == expected_standard_input)

    def test_get_single_groups_and_computes_input(self):
        sim = interface.Interface()
        types = ['1','4','56']
        groups_and_computes = sim.get_single_groups_and_computes_input(types)
        expected_groups_and_computes = [
            'group atom_type_1 type 1', 'group atom_type_4 type 4',
            'group atom_type_56 type 56', '\n#Atom Type 1 Computes',
            'compute type_1_energy atom_type_1 reduce sum c_per_atom',
            'variable number_of_type_1_atoms equal count(atom_type_1)',
            'variable atom_type_1_energy_per_atom equal '
            + 'c_type_1_energy/v_number_of_type_1_atoms',
            '\n#Atom Type 4 Computes',
            'compute type_4_energy atom_type_4 reduce sum c_per_atom',
            'variable number_of_type_4_atoms equal count(atom_type_4)',
            'variable atom_type_4_energy_per_atom equal '
            + 'c_type_4_energy/v_number_of_type_4_atoms',
            '\n#Atom Type 56 Computes',
            'compute type_56_energy atom_type_56 reduce sum c_per_atom',
            'variable number_of_type_56_atoms equal count(atom_type_56)',
            'variable atom_type_56_energy_per_atom equal '
            + 'c_type_56_energy/v_number_of_type_56_atoms']
        self.assertTrue(groups_and_computes == expected_groups_and_computes)

    def test_define_thermo_output_single(self):
        sim = interface.Interface()
        standard_input = ['#Thermo output style', '']
        types = ['2', '3']
        standard_input = sim.define_thermo_output_single(standard_input, types)
        expected_standard_input = ['#Thermo output style',
            ' c_type_2_energy v_number_of_type_2_atoms '
            + 'v_atom_type_2_energy_per_atom c_type_3_energy '
            + 'v_number_of_type_3_atoms v_atom_type_3_energy_per_atom']
        self.assertTrue(standard_input == expected_standard_input)

    def test_create_single_cell_files_for_interface_gives_correct_cells(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [3,0,0], [1,2,0],
                                            [-2,1,2])
        test_unitcell_2 = test_unitcell.copy()
        potential_types = [['Morse', 'Fe', 'Fe']]
        os.system('mkdir test_interface')
        sim.create_single_cell_files_for_interface('test_interface',
            test_unitcell, test_unitcell_2, potential_types)
        with open('test_interface/cell_1_only.in', 'r') as file:
            input_cell_1 = file.read()
        with open('test_interface/cell_2_only.in', 'r') as file:
            input_cell_2 = file.read()
        with open('test_files/cell_1_only_check.in', 'r') \
                as file:
            expected_input_cell_1 = file.read()[:-1]
        with open('test_files/cell_2_only_check.in', 'r') \
                as file:
            expected_input_cell_2 = file.read()[:-1]
        self.assertTrue(input_cell_1 == expected_input_cell_1)
        self.assertTrue(input_cell_2 == expected_input_cell_2)
        os.system('rm -r test_interface')

    def test_get_required_supercell_repeats_for_adequete_atoms(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Fe',0,0,0)]
        test_unitcell = ge.UnitCellExtended(test_basis, [2,0,0], [0,2,0],
            [0,0,2])
        repeat = sim.get_required_supercell_repeats_for_adequete_atoms(
            test_unitcell, 7000)
        expected_repeat = 19
        self.assertTrue(repeat == expected_repeat)

    def test_get_required_supercell_repeats_for_adequete_atoms_2(self):
        sim = interface.Interface()
        test_basis = [ge.Atom('Fe',0,0,0), ge.Atom('Nd',1,0,0),
                      ge.Atom('Zn',0.2,0.2,0.5)]
        test_unitcell = ge.UnitCellExtended(test_basis, [2,0,0], [0,2,0],
            [0,0,2])
        repeat = sim.get_required_supercell_repeats_for_adequete_atoms(
            test_unitcell, 12000)
        expected_repeat = 16
        self.assertTrue(repeat == expected_repeat)

    def OFF_test_refine_interface_atom_lists_returns_correct_lists(self):
        potential_types = [['Morse', 'Fe', 'Fe'], ['Morse', 'Fe', 'Nd'],
                           ['Morse', 'Nd', 'Nd']]
        interface_atom_lists = {
            'Full_List': [['Nd', 0.0, 0.0, 20.0], ['Nd', 2.0, 2.0, 22.401],
            ['Fe', 1.424, 0.0, 20.0], ['Fe', 3.4272, 2.0, 22.401],
            ['Fe', 2.5728, 0.0, 20.0], ['Fe', 0.5728, 2.0, 22.401]],
            'SuperCell_1': [['Nd', 0.0, 0.0, 20.0], ['Nd', 2.0, 2.0, 22.401],
            ['Fe', 1.424, 0.0, 20.0], ['Fe', 3.4272, 2.0, 22.401],
            ['Fe', 2.5728, 0.0, 20.0], ['Fe', 0.5728, 2.0, 22.401],
            ['Fe', 64, 64, 10.401], ['Nd', 15, 15, 15.401],
            ['Fe', 17, 20, 10.401]],
            'SuperCell_2': [['Nd', 2.34, 1.2, .098], ['Fe',18,19,1]]}
        test_basis = [ge.Atom('Fe', 0, 0, 0), ge.Atom('Fe', 0.5, 0.5, 0.5)]
        test_unitcell_1 = ge.UnitCellExtended(test_basis, [2,0,0], [0,2,0],
                          [0,0,2])
        test_unitcell_2 = test_unitcell_1.copy()
        sim = interface.Interface()
        types = ['1','2']
        interface_atom_lists = sim.refine_interface_atom_lists(
            interface_atom_lists, test_unitcell_1, test_unitcell_2,
            potential_types, types)
        expected_interface_atom_lists = {
            'SuperCell_1':
            [['Nd', 1.5272000000000001, 0.10000000000000009, 22.401],
            ['Fe', 2.9544, 0.10000000000000009, 22.401],
            ['Fe', 0.09999999999999998, 0.10000000000000009, 22.401],
            ['Fe', 63.5272, 62.1, 10.401], ['Nd', 14.5272, 13.1, 15.401],
            ['Fe', 16.5272, 18.1, 10.401]],
            'SuperCell_2': [['Nd', 1.8672, 0.30000000000000027, 2.098],
            ['Fe', 17.5272, 18.1, 3.0]], 'Full_List':
            [['Nd', 1.5272000000000001, 0.10000000000000009, 22.401],
            ['Fe', 2.9544, 0.10000000000000009, 22.401],
            ['Fe', 0.09999999999999998, 0.10000000000000009, 22.401],
            ['Fe', 63.5272, 62.1, 10.401], ['Nd', 14.5272, 13.1, 15.401],
            ['Fe', 16.5272, 18.1, 10.401],
            ['Nd', 1.8672, 0.30000000000000027, 2.098],
            ['Fe', 17.5272, 18.1, 3.0]]}
        self.assertTrue(interface_atom_lists == expected_interface_atom_lists)

    def test_get_max_unitcell_sizes_returns_correct_unitcell_size(self):
        test_basis = [ge.Atom('Fe', 0, 0, 0), ge.Atom('Fe', 0.5, 0.5, 0.5)]
        test_unitcell_1 = ge.UnitCellExtended(test_basis, [2,0,0], [0,2,0],
                          [0,0,2])
        test_unitcell_2 =  ge.UnitCellExtended(test_basis, [13,0,0], [0,2.4,0],
                          [0,0,1])
        sim = interface.Interface()
        cell_sizes = sim.get_max_unitcell_sizes(test_unitcell_1,
                                                test_unitcell_2)
        expected_cell_sizes = {
            'Max_X_Unitcell_Length': 13, 'Max_Y_Unitcell_Length': 2.4}
        self.assertTrue(cell_sizes == expected_cell_sizes)

    def test_get_interface_shift_pairs_gives_correct_list_of_pairs(self):
        cell_sizes = {'Max_X_Unitcell_Length': 3.83,
                      'Max_Y_Unitcell_Length': 3.83}
        sim = interface.Interface()
        test_basis = [
            ge.Atom('Fe', 0.0, 0.0, 0.0), ge.Atom('Fe', 0.5, 0.5, 0.0),
            ge.Atom('Pt', 0.5, 0.0, 0.5), ge.Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell_1 = ge.UnitCellExtended(test_basis, [3.83, 0, 0], [0,3.83,0],
                                         [0, 0, 3.711])
        shift_pairs = sim.get_interface_shift_pairs(cell_sizes, unitcell_1)
        self.assertTrue(len(shift_pairs) == 200)

    def test_get_z_shift_list_returns_correct_z_shifts(self):
        sim = interface.Interface()
        test_basis = [
            ge.Atom('Fe', 0.0, 0.0, 0.0), ge.Atom('Fe', 0.5, 0.5, 0.0),
            ge.Atom('Pt', 0.5, 0.0, 0.5), ge.Atom('Pt', 0.0, 0.5, 0.5)]
        unitcell_1 = ge.UnitCellExtended(test_basis, [3.83, 0, 0], [0,3.83,0],
                                         [0, 0, 3.711])
        z_shift_list = sim.get_z_shift_list(unitcell_1)
        expected_z_shift_list = [
            1.8555, 1.9976198286574127, 2.1397396573148257,
            2.2818594859722383, 2.4239793146296513, 2.566099143287064,
            2.6664780610385677, 2.708218971944477]
        self.assertTrue(z_shift_list == expected_z_shift_list)

    def test_get_minimisation_output(self):
        os.system('cp test_files/log_check.lammps ./log.lammps')
        sim = interface.Interface()
        minimisation_data = {}
        sim.get_minimisation_output(minimisation_data)
        expected_data = {
            'Interface_Energy': -0.60343593, 'Number_of_Interface_Atoms': 2,
            'Interface_Energy_Per_Atom': -0.30171796,
            'Interface_Energy_Type_1': -0.30171796, 'Number_of_Type_1_Atoms': 1,
            'Interface_Energy_Per_Type_1_Atom': -0.30171796,
            'Interface_Energy_Type_2': -0.30171796, 'Number_of_Type_2_Atoms': 1,
            'Interface_Energy_Per_Type_2_Atom': -0.30171796}
        self.assertTrue(minimisation_data == expected_data)
        os.system('rm log.lammps')

    def test_get_cut_off_returns_correct_cut_off(self):
        potential_types = [['Morse', 'Nd', 'Nd'],['Morse', 'Fe', 'Fe']]
        sim = interface.Interface()
        cut_off = sim.get_cut_off(potential_types)
        expected_cut_off = '12.0'
        self.assertTrue(cut_off == expected_cut_off)

    def test_change_simulation_box_size_gives_correct_box_change(self):
        standard_input = ['change_box all']
        box_bounds = {'X_Min': '0', 'X_Max': '12.32', 'Y_Min': '4.2',
                      'Y_Max': '98.0', 'Z_Min': '1.23', 'Z_Max': '3.456'}
        sim = interface.Interface()
        standard_input = sim.change_simulation_box_size(standard_input,
                                                        box_bounds)
        expected_standard_input = ['change_box all x final 0 12.32 y final '
                                   + '4.2 98.0 z final 1.23 3.456 units box']
        self.assertTrue(standard_input == expected_standard_input)

    def test_add_file_to_read_correctly_gives_read_data_command(self):
        read_number = 2
        standard_input = ['read_data #2']
        name = 'test_file'
        sim = interface.Interface()
        standard_input = sim.add_file_to_read(
            standard_input, read_number, name)
        expected_standard_input = ['read_data test_file_cell'
                                   + '_2.in group interface_2_atoms add append']
        self.assertTrue(standard_input == expected_standard_input)


    def test_define_interface_region_side_in_generates_correct_region(self):
        standard_input = ['region test_region block']
        region_name = 'test'
        cut_off = '12.0'
        sim = interface.Interface()
        standard_input = sim.define_interface_region(
            standard_input, region_name, cut_off)
        expected_standard_input = [
            'region test_region block $(xlo+12.0) $(xhi-12.0) $(ylo+12.0) '
            + '$(yhi-12.0) $(zlo+12.0) $(zhi-12.0)']
        self.assertTrue(standard_input == expected_standard_input)

    def test_get_atom_types_gets_the_correct_atoms(self):
        atom_list = [['Nd', 1, 2, 3], ['Fe', 1, 2, 3],
                           ['Zn', 1, 2, 3], ['Qt', 1, 2, 3],
                           ['Nd', 1, 2, 3], ['yoyoyoy', 1, 2, 3],
                           ['Nd', 1, 2, 3], ['He', 1, 2, 3]]
        sim = interface.Interface()
        types = sim.get_atom_types(atom_list)
        expected_types = ['1','2','3','4','5','6']
        self.assertTrue(types == expected_types)

    def test_define_thermo_output_gives_correct_thermo_commans(self):
        standard_input = ['v_number_of_unfrozen_interface_2_atoms '
                          + 'v_interface_2_energy_per_atom']
        sim = interface.Interface()
        types = ['1','2','5']
        standard_input = sim.define_thermo_output_interface(
            standard_input, types)
        with open('test_files/thermo_output_check.data', 'r') as file:
            expected_standard_input = [file.read()[:-1]]
        self.assertTrue(standard_input == expected_standard_input)

    def test_create_lammps_data_file_creates_correct_file(self):
        test_basis = [ge.Atom('Fe',0,0,0), ge.Atom('Fe',0.5,0.5,0),
                      ge.Atom('Pt',0.5,0,0.5), ge.Atom('Pt',0,0.5,0.5)]
        test_unitcell = ge.UnitCellExtended(test_basis, [3.83,0,0], [0,3.83,0],
                                            [0,0,3.711])
        test_supercell = ge.SuperCellExtended(test_unitcell,10,10,10)
        potential_types = [['Morse','Fe','Fe']]
        sim = interface.Interface()
        atom_list = test_supercell.get_cartesian_formatted_atom_list()
        atom_list = [atom[1:] for atom in atom_list]
        sim.create_lammps_data_file('lammps_data_file_test_interface',
            potential_types, atom_list)
        with open('lammps_data_file_test_interface.in', 'r') as file:
            data_file = file.read()
        with open('test_files/lammps_data_file_test_interface_check.in',
                'r') as file:
            expected_data_file = file.read()[:-1]
        self.assertTrue(data_file == expected_data_file)
        os.system('rm lammps_data_file_test_interface.in')

    def test_create_lammps_data_file_creates_correct_file_meam(self):
        test_basis = [ge.Atom('Fe',0,0,0), ge.Atom('Fe',0.5,0.5,0),
                      ge.Atom('Pt',0.5,0,0.5), ge.Atom('Pt',0,0.5,0.5)]
        test_unitcell = ge.UnitCellExtended(test_basis, [3.83,0,0], [0,3.83,0],
                                            [0,0,3.711])
        test_supercell = ge.SuperCellExtended(test_unitcell,10,10,10)
        potential_types = [['Meam/c LAMMPS','*','*']]
        sim = interface.Interface()
        atom_list = test_supercell.get_cartesian_formatted_atom_list()
        atom_list = [atom[1:] for atom in atom_list]
        sim.create_lammps_data_file('lammps_data_file_meam_test_interface',
            potential_types, atom_list)
        with open('lammps_data_file_meam_test_interface.in', 'r') as file:
            data_file = file.read()
        with open('test_files/lammps_data_file_meam_test_interface_check.in',
                'r') as file:
            expected_data_file = file.read()[:-1]
        self.assertTrue(data_file == expected_data_file)
        os.system('rm lammps_data_file_meam_test_interface.in')

    def test_format_atom_list_gives_correctly_formatted_atoms(self):
        element_dict = {'Nd': '1', 'Fe': '2'}
        atom_list = [['Nd', 0.0, 0.0, 20.0],
                     ['Nd', 2.0, 2.0, 22.401],
                     ['Fe', 1.424, 0.0, 20.0],
                     ['Fe', 3.4272, 2.0, 22.401],
                     ['Fe', 2.5728, 0.0, 20.0],
                     ['Fe', 0.5728, 2.0, 22.401]]
        sim = interface.Interface()
        atom_list = sim.format_atom_list_for_LAMMPS(atom_list, element_dict)
        expected_atom_list = ['1 1 0.0 0.0 20.0',
                              '2 1 2.0 2.0 22.401',
                              '3 2 1.424 0.0 20.0',
                              '4 2 3.4272 2.0 22.401',
                              '5 2 2.5728 0.0 20.0',
                              '6 2 0.5728 2.0 22.401']
        self.assertTrue(atom_list == expected_atom_list)

    def test_set_potential_style_meam_gives_correct_output(self):
        sim = interface.Interface()
        standard_input = ['pair_style', '#Pair_Coeff if MEAM']
        potential_types = [['Meam/c LAMMPS', '*', '*']]
        meam_files = {'Library': 'FePt.meam', 'Element_References':
                      ['Fe(L10)', 'Pt(L10)']}
        potential_input = sim.set_potential_style_meam(
            standard_input, potential_types, meam_files)
        expected_potential_input = ['pair_style meam/c','#Pair_Coeff if MEAM\n'
                                    + 'pair_coeff * * library.meam Fe(L10) '
                                    + 'Pt(L10) FePt.meam Fe(L10) Pt(L10)']
        self.assertTrue(potential_input == expected_potential_input)

    def test_get_atom_nearest_neighbours_gets_correct_neighbours(self):
        atom_list = [['Nd', 0.0, 0.0, 20.0], ['Nd', 2.0, 2.0, 22.401],
                     ['Fe', 1.424, 0.0, 20.0], ['Fe', 3.4272, 2.0, 22.401],
                     ['Fe', 2.5728, 0.0, 20.0], ['Fe', 0.5728, 2.0, 22.401]]
        atom = ['Nd', 0.0, 0.0, 20.0]
        sim = interface.Interface()
        neighbour_distances = sim.get_nearest_neighbours(atom, atom_list, 5)
        expected_neighbour_distances = (
            [0.0, 1.424, 2.5728, 3.1769326149605375, 3.7100944731906758])
        self.assertTrue(neighbour_distances == expected_neighbour_distances)


class TestAnalyseInterface(unittest.TestCase):

    def test_get_excel_spreadsheet_of_model_output_produces_spreadsheet(self):
        analyse = interface.AnalyseInterface()
        analyse.get_excel_spreadsheet_of_model_output(
            'test_files/interface_model_output.lammps')
        self.assertTrue(True)
        os.system('rm test_files/interface_model_output.xls')

    def OFF_test_get_interface_output_data(self):
        analyse = interface.AnalyseInterface()
        excel_data = analyse.get_interface_output_data(
            'test_files/interface_model_output.lammps')
        excel_data = str(excel_data)
        with open('test_files/excel_data_check.txt', 'r') as file:
            expected_data = file.read()[:-1]
        self.assertTrue(expected_data == excel_data)

    def test_get_interface_energy(self):
        analyse = interface.AnalyseInterface()
        interface_output_data = analyse.get_interface_output_data(
            'test_files/interface_model_output.lammps')
        analyse.get_data_averages(interface_output_data, 3)
        analyse.get_interface_energy(interface_output_data)
        expected_energy = -147.99924874899443
        self.assertTrue(
            interface_output_data['Interface_Energy'] == expected_energy)
        expected_energy = -0.004849235265212811
        self.assertTrue(
            interface_output_data['Interface_Energy/Area'] == expected_energy)

    def test_get_data_averages_averages_data_correctly(self):
        analyse = interface.AnalyseInterface()
        test_data = {'Run_1': {'Headers': ['Header_1', 'Header_2', 'Header_3'],
                     'Data': [[1,3,5],[1.2,2.8,5],[-1,2,5]]}}
        analyse.get_data_averages(test_data, 0)
        expected_data = {'Run_1': {'Headers':
                         ['Header_1', 'Header_2', 'Header_3'],
                         'Data': [[1.0,3.0,5.0],[1.2,2.8,5.0],[-1.0,2.0,5.0]],
                         'Averaged_Data': {'Header_1': 0.4,
                         'Header_2': 2.6, 'Header_3': 5.0}}}
        self.assertTrue(expected_data == test_data)
        analyse.get_data_averages(test_data, 1)
        expected_data = {'Run_1': {'Headers':
                         ['Header_1', 'Header_2', 'Header_3'],
                         'Data': [[1,3,5],[1.2,2.8,5],[-1,2,5]],
                         'Averaged_Data': {'Header_1': 0.1, 'Header_2': 2.4,
                         'Header_3': 5.0}}}
        self.assertTrue(test_data == expected_data)





if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory)
    import grain_extended as ge
    import interface
    unittest.main()
