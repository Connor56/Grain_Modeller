import grain_extended as ge
import interface
import os
import random

class InterfaceRandomise(interface.Interface):

    def supercell_randomise(self, supercell, ratios):
        '''
        Takes in a supercell and randomises it based on the elements already
        present in it and the ratio list given to it. The ratios should
        add to 1, and they should be in alpahbetical order, i.e. if you have
        3 elements (Fe, Zn, Pt), then you should rearrange them alphabetically
        in your mind to (Fe, Pt, Zn), then have their desired ratios in the
        same order. Therefore, with elements (Fe, Zn, Pt), a ratio list of
        [0.1, 0.3, 0.6] will result in: Fe=0.1, Pt=0.3, Zn=0.6. This will
        return a randomised atom structure with roughly the right amount of
        atoms of each type.

        supercell: SupercellExtended object that you wish to randomise.
        ratios: List of floats adding to one, representing the desired ratios
            of the different elements, for information on how this works,
            please read above.
        return: Nothing, although the supercell given to it is changed.
        '''
        supercell_atoms = supercell.get_cartesian_formatted_atom_list()
        supercell_atoms = list(set([atom[1] for atom in supercell_atoms]))
        supercell_atoms.sort()
        if sum(ratios) > 1:
            raise ValueError('Summation of the Ratios is greater than 1. '
                             + 'Change ratios so that they sum to 1, and '
                             + 'try again.')
        print('Caution: Ratios should be input in the order cell elements '
              + 'appear alphabetically. E.g. if you have iron: Fe, and '
              + 'Platinum: Pt, then the ratio order should be: [Fe Ratio, '
              + 'Pt Ratio] in the ratio list, because F is before P '
              + 'alphabetically.\n\nYour ratio inputs are: ')
        try:
            for index in range(len(supercell_atoms)):
                print(supercell_atoms[index]+':', ratios[index])
        except IndexError:
            raise IndexError('Number of ratios does not match the number of'
                             + 'atoms types in the supercell')
        element_ratios = {}
        ratio_sum = 0
        for index in range(len(supercell_atoms)):
            element_ratios[supercell_atoms[index]] = {
                'Ratio_Start': ratio_sum,
                'Ratio_End': ratio_sum + ratios[index]}
            ratio_sum += ratios[index]
        for atom in supercell.get_atoms():
            random_value = random.randint(0,9999)/10000
            atom_type = ''
            for key, item in element_ratios.items():
                if item['Ratio_Start'] <= random_value < item['Ratio_End']:
                    atom_type = key
            atom.element = atom_type

    def create_single_cell_files_for_interface(self, name, unitcell_1,
            unitcell_2, potential_types, randomise, ratios):
        cell_repeat = self.get_required_supercell_repeats_for_adequete_atoms(
            unitcell_1, 8000)
        supercell_1 = ge.SuperCellExtended(
            unitcell_1, cell_repeat, cell_repeat, cell_repeat)
        if randomise['Cell_1'] == True:
            self.supercell_randomise(supercell_1, ratios['Cell_1'])
        cell_repeat = self.get_required_supercell_repeats_for_adequete_atoms(
            unitcell_2, 8000)
        supercell_2 = ge.SuperCellExtended(
            unitcell_2, cell_repeat, cell_repeat, cell_repeat)
        if randomise['Cell_2'] == True:
            self.supercell_randomise(supercell_2, ratios['Cell_2'])
        supercell_1.create_lammps_data_file(
            name+'/cell_1_only', potential_types)
        supercell_2.create_lammps_data_file(
            name+'/cell_2_only', potential_types)
        end_name = name.split('/')[-1]

    def create_interface_simulation(self, name, unitcell_1, unitcell_2,
            max_side_length, potential_types,
            randomise = {'Cell_1': False, 'Cell_2': False},
            ratios = {'Cell_1': [], 'Cell_2': []}, meam_files = None):
        repeat = self.match_cells(unitcell_1, unitcell_2, max_side_length)
        repeat = self.match_repeats_to_max_side_length(
                 unitcell_1, repeat, max_side_length)
        interface_atom_lists = self.create_interface_atom_list(
            unitcell_1, unitcell_2, repeat, randomise, ratios, 45)
        types = self.get_atom_types(interface_atom_lists['Full_List'])
        interface_atom_lists = self.refine_interface_atom_lists(
            interface_atom_lists, unitcell_1, unitcell_2, potential_types,
            types, meam_files)
        os.system('mkdir '+name)
        self.create_lammps_data_file(name+'/'+name+'_cell_1', potential_types,
                interface_atom_lists['SuperCell_1'])
        self.create_lammps_data_file(name+'/'+name+'_cell_2', potential_types,
                interface_atom_lists['SuperCell_2'])
        box_bounds = self.get_box_bounds_interface(
                    interface_atom_lists['SuperCell_1']+
                    interface_atom_lists['SuperCell_2'])
        self.create_interface_input(
            box_bounds, name, potential_types, types, meam_files)
        self.create_standard_lammps_input(
            name, unitcell_1, unitcell_2, potential_types, types, randomise,
            ratios, meam_files)

    def create_interface_atom_list(self, unitcell_1, unitcell_2, repeat,
            randomise, ratios, z_size = 40):
        z_repeat_1 = self.get_z_repeat_for_unitcell(unitcell_1, z_size)
        supercell_1 = ge.SuperCellExtended(unitcell_1,
                      repeat['Cell_1_X_Repeat'], repeat['Cell_1_Y_Repeat'],
                      z_repeat_1)
        if randomise['Cell_1'] == True:
            self.supercell_randomise(supercell_1, ratios['Cell_1'])
        z_repeat_2 = self.get_z_repeat_for_unitcell(unitcell_2, z_size)
        supercell_2 = ge.SuperCellExtended(unitcell_2,
                      repeat['Cell_2_X_Repeat'], repeat['Cell_2_Y_Repeat'],
                      z_repeat_2)
        if randomise['Cell_2'] == True:
            self.supercell_randomise(supercell_2, ratios['Cell_2'])
        supercell_atom_lists = self.stack_supercells(supercell_1, supercell_2)
        supercell_1_attributes = supercell_1.get_attributes()
        supercell_2_attributes = supercell_2.get_attributes()
        interface_bounds = self.get_interface_bounds(supercell_1_attributes,
                           supercell_2_attributes)
        supercell_atom_lists = self.format_supercell_atom_lists(
                               supercell_atom_lists)
        supercell_atom_lists = self.trim_supercell_atoms(
                               supercell_atom_lists['SuperCell_1'],
                               supercell_atom_lists['SuperCell_2'],
                               interface_bounds)
        return supercell_atom_lists

    def create_standard_lammps_input(self, name, unitcell_1, unitcell_2,
            potential_types, types, randomise, ratios, meam_files = None):
        self.create_single_cell_files_for_interface(
            name, unitcell_1, unitcell_2, potential_types, randomise, ratios)
        standard_input = ['\n\nclear\n\n']
        standard_input += self.create_input_for_single_LAMMPS_simulation(
            name+'_cell_1_only', 1, potential_types, types, meam_files)
        standard_input += ['\n\nclear\n\n']
        standard_input += self.create_input_for_single_LAMMPS_simulation(
            name+'_cell_2_only', 2, potential_types, types, meam_files)
        with open(name+'/'+name+'.interface', 'a+') as file:
            file.write('\n'.join(standard_input))

    def get_gulp_minimised_unitcell_for_randomised_structure(
            self, potential_types, unitcell, ratios):
        required_repeats = (
            self.get_required_supercell_repeats_for_adequete_atoms(
                unitcell, 4000))
        supercell = ge.SuperCellExtended(unitcell, required_repeats,
            required_repeats, required_repeats)
        self.supercell_randomise(supercell, ratios)
        self.run_in_gulp()
