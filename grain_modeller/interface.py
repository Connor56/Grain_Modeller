import grain_extended as ge
import os
import xlwt
import numpy as np
from scipy import spatial

element_weight_dictionary = {'Fe':{'Mass': 55.85},'Pt':{'Mass': 195.08},
                             'Nd':{'Mass': 144.24},'Ti':{'Mass': 47.87}}

class Interface():

    def get_correct_unitcell(self, unitcell):
        supercell = ge.SuperCellExtended(unitcell['Cell'],8,8,8)
        new_unitcell = supercell.define_new_cubic_unitcell(unitcell['Normal'],
                       [2,2,2], unitcell['Potentials'], unitcell['Simulation'])
        return new_unitcell

    def match_cells(self, unitcell1, unitcell2, max_xy_length):
        if (not all([type(unitcell1) == ge.UnitCellExtended or
                     type(unitcell1) == ge.UnitCellExtendedFile,
                     type(unitcell2) == ge.UnitCellExtended or
                     type(unitcell2) == ge.UnitCellExtendedFile])):
            raise TypeError('Unitcells must be of type: UnitCellExtended or '+
                            'UnitCellExtendedFile.')
        cell1_attributes = unitcell1.get_attributes()
        cell2_attributes = unitcell2.get_attributes()
        cell_1_x = cell1_attributes['a_Lattice_Vector'][0]
        cell_1_y = cell1_attributes['b_Lattice_Vector'][1]
        cell_2_x = cell2_attributes['a_Lattice_Vector'][0]
        cell_2_y = cell2_attributes['b_Lattice_Vector'][1]
        ratio_x = cell_1_x/cell_2_x
        ratio_y = cell_1_y/cell_2_y
        ratio_list_x = [ [1*repeat, ratio_x*repeat, cell_1_x*repeat,
                          cell_2_x*repeat*ratio_x] for repeat in range(1, 100)]
        ratio_list_y = [ [1*repeat, ratio_y*repeat, cell_1_y*repeat,
                          cell_2_y*repeat*ratio_y] for repeat in range(1, 100)]
        ratio_list_x = [ ratio_item+[abs(ratio_item[1]-round(ratio_item[1]))]
                         for ratio_item in ratio_list_x if
                         ratio_item[2]+ratio_item[3] < 2*max_xy_length]
        ratio_list_y = [ ratio_item+[abs(ratio_item[1]-round(ratio_item[1]))]
                         for ratio_item in ratio_list_y if
                         ratio_item[2]+ratio_item[3] < 2*max_xy_length]
        ratio_list_x.sort(key = lambda x: (x[-1], x[2]+x[3]))
        ratio_list_y.sort(key = lambda x: (x[-1], x[2]+x[3]))
        return_ratio = {'Cell_1_X_Repeat': ratio_list_x[0][0],
                        'Cell_1_Y_Repeat': ratio_list_y[0][0],
                        'Cell_2_X_Repeat': int(round(ratio_list_x[0][1])),
                        'Cell_2_Y_Repeat': int(round(ratio_list_y[0][1]))}
        return return_ratio

    def create_interface_atom_list(self, unitcell_1, unitcell_2, repeat,
                                   z_size = 40):
        z_repeat_1 = self.get_z_repeat_for_unitcell(unitcell_1, z_size)
        supercell_1 = ge.SuperCellExtended(unitcell_1,
                      repeat['Cell_1_X_Repeat'], repeat['Cell_1_Y_Repeat'],
                      z_repeat_1)
        z_repeat_2 = self.get_z_repeat_for_unitcell(unitcell_2, z_size)
        supercell_2 = ge.SuperCellExtended(unitcell_2,
                      repeat['Cell_2_X_Repeat'], repeat['Cell_2_Y_Repeat'],
                      z_repeat_2)
        supercell_atom_lists = self.stack_supercells(supercell_1, supercell_2)
        supercell_1_attributes = supercell_1.get_attributes()
        supercell_2_attributes = supercell_2.get_attributes()
        interface_bounds = self.get_interface_bounds(
            supercell_1_attributes, supercell_2_attributes)
        supercell_atom_lists = self.format_supercell_atom_lists(
            supercell_atom_lists)
        supercell_atom_lists = self.trim_supercell_atoms(
                               supercell_atom_lists['SuperCell_1'],
                               supercell_atom_lists['SuperCell_2'],
                               interface_bounds)
        return supercell_atom_lists

    def get_z_repeat_for_unitcell(self, unitcell, z_size):
        unitcell_attributes = unitcell.get_attributes()
        z_height = unitcell_attributes['c_Lattice_Vector'][2]
        z_repeat = int(round(z_size/z_height))
        return(z_repeat)

    def stack_supercells(self, supercell_1, supercell_2):
        supercell_1_atom_list = supercell_1.get_cartesian_formatted_atom_list()
        supercell_1_height = self.get_box_bounds_interface(
            [atom[1:] for atom in supercell_1_atom_list])['Z_Max']
        supercell_2_atom_list = supercell_2.get_cartesian_formatted_atom_list()
        supercell_2_atom_list = [atom[:4]+[atom[4]+supercell_1_height] for
                                 atom in supercell_2_atom_list]
        return {'SuperCell_1': supercell_1_atom_list,
                'SuperCell_2': supercell_2_atom_list,
                'Full_List': supercell_1_atom_list + supercell_2_atom_list}

    def trim_supercell_atoms(self, supercell_1_atom_list, supercell_2_atom_list,
                             interface_bounds):
        supercell_1_atom_list = [
                    atom for atom in supercell_1_atom_list if
                    interface_bounds['X_Min'] < atom[1] <
                    interface_bounds['X_Max'] and
                    interface_bounds['Y_Min'] < atom[2] <
                    interface_bounds['Y_Max']]
        supercell_2_atom_list = [
                    atom for atom in supercell_2_atom_list if
                    interface_bounds['X_Min'] < atom[1] <
                    interface_bounds['X_Max'] and
                    interface_bounds['Y_Min'] < atom[2] <
                    interface_bounds['Y_Max']]
        return {'SuperCell_1': supercell_1_atom_list,
                'SuperCell_2': supercell_2_atom_list,
                'Full_List': supercell_1_atom_list + supercell_2_atom_list}

    def format_supercell_atom_lists(self, supercell_atom_lists):
        supercell_atom_lists['Full_List'].sort(
            key = lambda x: (x[1], x[2], x[3], x[4]))
        supercell_atom_lists['Full_List'] = [ atom[1:] for atom in
            supercell_atom_lists['Full_List']]
        supercell_atom_lists['SuperCell_1'] = [ atom[1:] for atom in
            supercell_atom_lists['SuperCell_1']]
        supercell_atom_lists['SuperCell_1'].sort(
            key = lambda x: (x[0], x[1], x[2], x[3]))
        supercell_atom_lists['SuperCell_2'] = [ atom[1:] for atom in
            supercell_atom_lists['SuperCell_2']]
        supercell_atom_lists['SuperCell_2'].sort(
            key = lambda x: (x[0], x[1], x[2], x[3]))
        return supercell_atom_lists

    def create_interface_simulation(self, name, unitcell_1, unitcell_2,
            max_side_length, potential_types, meam_files = None):
        repeat = self.match_cells(unitcell_1, unitcell_2, max_side_length)
        repeat = self.match_repeats_to_max_side_length(
                 unitcell_1, repeat, max_side_length)
        interface_atom_lists = self.create_interface_atom_list(
            unitcell_1, unitcell_2, repeat, 45)
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
        box_bounds['X_Max'] += 2; box_bounds['X_Min'] -= 2
        box_bounds['Y_Max'] += 2; box_bounds['Y_Min'] -= 2
        box_bounds['Z_Max'] += 2; box_bounds['Z_Min'] -= 2
        self.create_interface_input(
            box_bounds, name, potential_types, types, meam_files)
        self.create_standard_lammps_input(
            name, unitcell_1, unitcell_2, potential_types, types, meam_files)

    def shift_supercell(self, supercell_atom_list, shifts):
        supercell_atom_list_new = [
            [atom[0], atom[1]+shifts['X_Shift'], atom[2]+shifts['Y_Shift'],
            atom[3]+shifts['Z_Shift']] for atom in supercell_atom_list]
        return supercell_atom_list_new

    def realign_interface(self, interface_atom_lists, box_bounds):
        interface_atom_lists['SuperCell_1'] = [ [atom[0],
                    atom[1]-box_bounds['X_Min'],
                    atom[2]-box_bounds['Y_Min'],
                    atom[3]] for atom in interface_atom_lists['SuperCell_1']]
        interface_atom_lists['SuperCell_2'] = [ [atom[0],
                    atom[1]-box_bounds['X_Min'],
                    atom[2]-box_bounds['Y_Min'],
                    atom[3]] for atom in interface_atom_lists['SuperCell_2']]
        interface_atom_lists['Full_List'] = (
            interface_atom_lists['SuperCell_1']+
            interface_atom_lists['SuperCell_2'])
        return interface_atom_lists

    def match_repeats_to_max_side_length(
            self, unitcell_1, repeat, max_side_length):
        x_repeat = int((max_side_length)/(repeat['Cell_1_X_Repeat']*
                   unitcell_1.get_attributes()['a_Lattice_Vector'][0]))
        if x_repeat > 1:
            repeat['Cell_1_X_Repeat'] = repeat['Cell_1_X_Repeat']*x_repeat
            repeat['Cell_2_X_Repeat'] = repeat['Cell_2_X_Repeat']*x_repeat
        y_repeat = int((max_side_length)/(repeat['Cell_1_Y_Repeat']*
                   unitcell_1.get_attributes()['b_Lattice_Vector'][1]))
        if y_repeat > 1:
            repeat['Cell_1_Y_Repeat'] = repeat['Cell_1_Y_Repeat']*y_repeat
            repeat['Cell_2_Y_Repeat'] = repeat['Cell_2_Y_Repeat']*y_repeat
        return repeat

    def refine_interface_atom_lists(self, interface_atom_lists, unitcell_1,
            unitcell_2, potential_types, types, meam_files = None):
        boundary_shifts = self.orientate_boundary(
            interface_atom_lists, unitcell_1, unitcell_2, potential_types,
            types, meam_files)
        box_bounds = self.get_box_bounds_interface(
                     interface_atom_lists['Full_List'])
        box_bounds['X_Min'] = box_bounds['X_Min']+boundary_shifts['X_Shift']
        box_bounds['Y_Min'] = box_bounds['Y_Min']+boundary_shifts['Y_Shift']
        interface_atom_lists['SuperCell_2'] = self.shift_supercell(
                    interface_atom_lists['SuperCell_2'], boundary_shifts)
        interface_atom_lists = self.trim_supercell_atoms(
                               interface_atom_lists['SuperCell_1'],
                               interface_atom_lists['SuperCell_2'],
                               box_bounds)
        box_bounds = self.get_box_bounds_interface(
                     interface_atom_lists['Full_List'])
        interface_atom_lists = self.realign_interface(
                               interface_atom_lists, box_bounds)
        return interface_atom_lists

    def create_xyz_file(self, file_name, atom_list):
        atom_print_list = [[atom[0]]+list(map(float, atom[1:])) for
                           atom in atom_list]
        with open(file_name+'.xyz', 'w') as write_file:
            number_of_atoms = len(atom_print_list)
            atom_print_list = [ [atom[0]]+list(map("{:19.9f}".format, atom[1:]))
                                for atom in atom_print_list]
            print_string = str(number_of_atoms)+'\n\n'
            print_string += (
                    '\n'.join([ ' '.join(atom) for atom in atom_print_list]))
            write_file.write(print_string)

    def create_lammps_data_file(self, name, potential_types, atom_list):
        setup = ge.Setup()
        element_list = list(set([ atom[0] for atom in atom_list]))
        element_list.sort()
        element_masses = setup.get_element_masses(element_list)
        box_bounds = self.get_box_bounds_interface(atom_list)
        box_bounds['X_Max'] += 2; box_bounds['X_Min'] -= 2
        box_bounds['Y_Max'] += 2; box_bounds['Y_Min'] -= 2
        box_bounds['Z_Max'] += 2; box_bounds['Z_Min'] -= 2
        for key, item in box_bounds.items():
            box_bounds[key] = str(item)
        number_of_atoms = str(len(atom_list))
        number_of_atom_types = str(len(element_list))
        element_dict = { element_list[type-1]: str(type) for type in range(1,
                        len(element_list)+1)}
        print_list = [name,'',number_of_atoms+' atoms',number_of_atom_types+
                      ' atom types', box_bounds['X_Min']+' '
                      + box_bounds['X_Max']+' xlo xhi',
                      box_bounds['Y_Min']+' '+box_bounds['Y_Max']+' ylo yhi',
                      box_bounds['Z_Min']+' '+box_bounds['Z_Max']+' zlo zhi',
                      box_bounds['XY_Length']+' '+ box_bounds['XZ_Length'] +' '
                      + box_bounds['YZ_Length']+' xy xz yz','','Atoms','']
        print_list += self.format_atom_list_for_LAMMPS(atom_list, element_dict)
        print_list += ['','Masses','']
        mass_list = [ [element_dict[element], element_masses[element]]
                     for element in element_list]
        mass_list = [ ' '.join(mass) for mass in mass_list]
        print_list += mass_list
        if potential_types[0][0] != 'Meam/c LAMMPS':
            potential_input = setup.get_potential_input(potential_types)
            potential_input = self.format_potential_input_for_LAMMPS(
                potential_input, element_dict)
            print_list += ['','PairIJ Coeffs','']
            potential_print = [ ' '.join(potential[1:]) for potential in
                               potential_input]
            print_list += potential_print
        setup.create_input_file(print_list, name, '.in')

    def format_potential_input_for_LAMMPS(self, potential_input, element_dict):
        potential_input = [[potential[0],element_dict[potential[1]],
                            element_dict[potential[2]],
                            potential[3]] for potential in potential_input]
        potential_input = [[potential[0], potential[1], potential[2],
                            potential[3]] if int(potential[1]) <
                            int(potential[2]) else [potential[0], potential[2],
                            potential[1], potential[3]]
                            for potential in potential_input]
        return potential_input

    def format_atom_list_for_LAMMPS(self, atom_list, element_dict):
        atom_list = [[str(count+1), element_dict[atom_list[count][0]]]+
                     list(map(str, atom_list[count][1:])) for count in
                     range(len(atom_list))]
        atom_list = [ ' '.join(atom) for atom in atom_list]
        return atom_list

    def get_interface_bounds(self, supercell_1_attributes,
                             supercell_2_attributes):
        attribute_list = [supercell_1_attributes, supercell_2_attributes]
        interface_bounds = []
        counter = 1
        for supercell in attribute_list:
            x_lower_bound = 0
            x_upper_bound = supercell['a_Lattice_Vector'][0]
            y_lower_bound = 0
            y_upper_bound = supercell['b_Lattice_Vector'][1]
            if supercell['b_Lattice_Vector'][0] < 0:
                x_upper_bound += supercell['b_Lattice_Vector'][0]
            elif supercell['b_Lattice_Vector'][0] > 0:
                x_lower_bound += supercell['b_Lattice_Vector'][0]
            if supercell['c_Lattice_Vector'][0] < 0:
                x_upper_bound += supercell['c_Lattice_Vector'][0]
            elif supercell['c_Lattice_Vector'][0] > 0:
                x_lower_bound += supercell['c_Lattice_Vector'][0]
            if supercell['c_Lattice_Vector'][1] < 0:
                y_upper_bound += supercell['c_Lattice_Vector'][1]
            elif supercell['c_Lattice_Vector'][1] > 0:
                y_lower_bound += supercell['c_Lattice_Vector'][1]
            interface_bounds.append({
                'X_Lower_Bound': x_lower_bound,'X_Upper_Bound': x_upper_bound,
                'Y_Lower_Bound': y_lower_bound,'Y_Upper_Bound': y_upper_bound })
        interface_bounds.sort(key = lambda x: x['X_Lower_Bound'])
        x_lower_bound = interface_bounds[-1]['X_Lower_Bound']-0.0001
        interface_bounds.sort(key = lambda x: x['X_Upper_Bound'])
        x_upper_bound = interface_bounds[0]['X_Upper_Bound']+0.0001
        interface_bounds.sort(key = lambda x: x['Y_Lower_Bound'])
        y_lower_bound = interface_bounds[-1]['Y_Lower_Bound']-0.0001
        interface_bounds.sort(key = lambda x: x['Y_Upper_Bound'])
        y_upper_bound = interface_bounds[0]['Y_Upper_Bound']+0.0001
        return {'X_Min': x_lower_bound,'X_Max': x_upper_bound,
                'Y_Min': y_lower_bound,'Y_Max': y_upper_bound }

    def get_box_bounds_interface(self, atom_list):
        x_coords = [atom[1] for atom in atom_list]
        x_coords.sort()
        x_min = x_coords[0]-0.1
        x_max = x_coords[-1]+0.1
        y_coords = [atom[2] for atom in atom_list]
        y_coords.sort()
        y_min = y_coords[0]-0.1
        y_max = y_coords[-1]+0.1
        z_coords = [atom[3] for atom in atom_list]
        z_coords.sort()
        z_min = z_coords[0]
        z_max = z_coords[-1]
        box_bounds = {'X_Min': x_min, 'X_Max': x_max, 'Y_Min': y_min,
                      'Y_Max': y_max, 'Z_Min': z_min, 'Z_Max': z_max,
                      'XY_Length': '0', 'XZ_Length': '0', 'YZ_Length': '0'}
        return box_bounds

    def orientate_boundary(self, interface_atom_lists,
            unitcell_1, unitcell_2, potential_types, types, meam_files = None):
        interface_atom_lists['SuperCell_2'] = [
                    atom[:3]+[atom[-1]] for atom in
                    interface_atom_lists['SuperCell_2']]
        interface_atom_lists['Full_List'] = (
                    interface_atom_lists['SuperCell_1']
                    + interface_atom_lists['SuperCell_2'])
        box_bounds = self.get_box_bounds_interface(
            interface_atom_lists['Full_List'])
        cell_sizes = self.get_max_unitcell_sizes(unitcell_1, unitcell_2)
        shift_pairs = self.get_interface_shift_pairs(
            cell_sizes, unitcell_1)
        cut_off = 14
        for shift in shift_pairs:
            print(shift)
            atom_list_2_shifted = self.shift_supercell(
                interface_atom_lists['SuperCell_2'], shift)
            atom_list = interface_atom_lists['SuperCell_1']+atom_list_2_shifted
            self.create_lammps_data_file('boundary_orientation',
                                         potential_types, atom_list)
            self.create_orientation_input(
                potential_types, box_bounds, cut_off, cell_sizes, types,
                meam_files)
            os.system('mpirun -np 2 ./lmp_mpi -in orientate_boundary.min')
            self.get_minimisation_output(shift)
            os.system('rm log.lammps boundary_orientation.in '
                      + 'orientation_minimize.xyz orientate_boundary.min')
        shift_pairs.sort(key = lambda x: x['Interface_Energy_Per_Atom'])
        print([[shift['Interface_Energy_Per_Atom'], shift['X_Shift'],
                shift['Y_Shift'], shift['Z_Shift']] for shift in shift_pairs])
        print('The chosen shift is: '+str(shift_pairs[0]))
        return(shift_pairs[0])

    def get_max_unitcell_sizes(self, unitcell_1, unitcell_2):
        unitcell_1_attributes = unitcell_1.get_attributes()
        unitcell_2_attributes = unitcell_2.get_attributes()
        cell_sizes = {'Max_X_Unitcell_Length': 0, 'Max_Y_Unitcell_Length': 0}
        if (unitcell_1_attributes['a_Lattice_Vector'][0] <
           unitcell_2_attributes['a_Lattice_Vector'][0]):
            cell_sizes['Max_X_Unitcell_Length'] = unitcell_2_attributes[
                                                  'a_Lattice_Vector'][0]
        else:
            cell_sizes['Max_X_Unitcell_Length'] = unitcell_1_attributes[
                                                  'a_Lattice_Vector'][0]
        if (unitcell_1_attributes['b_Lattice_Vector'][1] <
           unitcell_2_attributes['b_Lattice_Vector'][1]):
            cell_sizes['Max_Y_Unitcell_Length'] = unitcell_2_attributes[
                                                  'b_Lattice_Vector'][1]
        else:
            cell_sizes['Max_Y_Unitcell_Length'] = unitcell_1_attributes[
                                                  'b_Lattice_Vector'][1]
        return cell_sizes

    def get_interface_shift_pairs(self, cell_sizes, unitcell_1):
        x_shift_list = [count*0.5 for count in range(int(round(
                        cell_sizes['Max_X_Unitcell_Length']))+1)]
        y_shift_list = [count*0.5 for count in range(int(round(
                        cell_sizes['Max_Y_Unitcell_Length']))+1)]
        z_shift_list = self.get_z_shift_list(unitcell_1)
        shift_pairs = [
            {'X_Shift': x_shift,'Y_Shift': y_shift, 'Z_Shift': z_shift}
            for x_shift in x_shift_list for y_shift in y_shift_list
            for z_shift in z_shift_list]
        return shift_pairs

    def get_z_shift_list(self, unitcell):
        supercell_1 = ge.SuperCellExtended(unitcell, 2, 2, 2)
        supercell_atoms = supercell_1.get_cartesian_formatted_atom_list()
        supercell_atoms = [atom[1:] for atom in supercell_atoms]
        neighbour_distances = self.get_nearest_neighbours(
            supercell_atoms[0], supercell_atoms[1:], len(supercell_atoms)-1)
        neighbour_distances = list(set(neighbour_distances))
        neighbour_distances.sort()
        neighbour_distances = neighbour_distances[:2]
        z_separations = [
            abs(supercell_atoms[0][3]-atom[3]) for atom in supercell_atoms[1:]
            if abs(supercell_atoms[0][3]-atom[3]) != 0.0]
        z_separations = list(set(z_separations))[:2]
        z_separations = [separation for separation in z_separations if
                         separation < neighbour_distances[1]]
        z_shift_list = neighbour_distances+z_separations
        z_shift_list.sort()
        shift_range = z_shift_list[-1]-z_shift_list[0]
        number_of_steps = int(round(shift_range/0.15))
        range_step = shift_range/number_of_steps
        range_list = [z_shift_list[0]+(range_step*step) for step in
                      range(number_of_steps)]
        z_shift_list += range_list
        z_shift_list = list(set(z_shift_list))
        z_shift_list.sort()
        trimmed_z_shift_list = []
        definitely_keep_list = neighbour_distances+z_separations
        discard_list = []
        for index in range(len(z_shift_list)-1):
            if (z_shift_list[index+1] - z_shift_list[index]) < range_step/2:
                if z_shift_list[index] in definitely_keep_list:
                    trimmed_z_shift_list.append(z_shift_list[index])
                if not z_shift_list[index+1] in definitely_keep_list:
                    discard_list.append(z_shift_list[index+1])
            elif z_shift_list[index] not in discard_list:
                trimmed_z_shift_list.append(z_shift_list[index])
        trimmed_z_shift_list.append(z_shift_list[-1])
        return trimmed_z_shift_list

    def get_minimisation_output(self, shift):
        with open('log.lammps', 'r') as file:
            data = file.read().split('\n')
            data_index = [ index for index in range(len(data)) if
                          data[index][:9] == 'Loop time'][0]-1
            final_data_line = [ item for item in
                                data[data_index].split(' ') if item != '']
            shift['Interface_Energy'] = float(final_data_line[6])
            shift['Number_of_Interface_Atoms'] = int(final_data_line[7])
            shift['Interface_Energy_Per_Atom'] = float(final_data_line[8])
            type_range = int(len(final_data_line[9:])/3)
            types = [ type for type in range(1, type_range+1)]
            for type in types:
                shift['Interface_Energy_Type_'+str(type)] = (
                        float(final_data_line[9+((type-1)*3)]))
                shift['Number_of_Type_'+str(type)+'_Atoms'] = (
                        int(final_data_line[10+((type-1)*3)]))
                shift['Interface_Energy_Per_Type_'+str(type)+'_Atom'] = (
                        float(final_data_line[11+((type-1)*3)]))

    def create_orientation_input(self, potential_types, box_bounds, cut_off,
            cell_sizes, types, meam_files = None):
        setup = ge.Setup()
        standard_input = setup.get_standard_input(
            'Boundary_Orientation_Interface')
        if potential_types[0][0] == 'Meam/c LAMMPS':
            stanard_input = self.set_potential_style_meam(
                standard_input, potential_types, meam_files)
        else:
            standard_input = self.set_potential_style(
                standard_input, potential_types)
        region_index = standard_input.index(
            'region check_interface_region block')
        standard_input[region_index] += (' $(xlo+'+str(
            cell_sizes['Max_X_Unitcell_Length']/2)+'+'+str(cut_off)+') '
            + '$(xhi-'+str(cell_sizes['Max_X_Unitcell_Length']/2)+'-'
            + str(cut_off)+') '
            + '$(ylo+'+str(cell_sizes['Max_Y_Unitcell_Length']/2)+'+'
            + str(cut_off)+') '
            + '$(yhi-'+str(cell_sizes['Max_Y_Unitcell_Length']/2)+'-'
            + str(cut_off)+') EDGE EDGE')
        groups_and_computes = self.get_groups_and_computes_input(
                              types, ['interface'], 'check')
        atom_type_compute_index = standard_input.index('#Atom Type Computes')
        standard_input = (standard_input[:atom_type_compute_index+1]+
                          groups_and_computes+
                          standard_input[atom_type_compute_index+1:])
        thermo_index = standard_input.index('thermo_style custom step temp '+
                    'epair emol etotal press c_interface_energy '+
                    'v_number_of_interface_atoms v_interface_energy_per_atom')
        for type in types:
            standard_input[thermo_index] += (' c_type_'+type
                + '_interface_energy'+' v_number_of_type_'+type
                + '_interface_atoms '
                + 'v_interface_type_'+type+'_energy_per_atom')
        setup.create_input_file(standard_input, 'orientate_boundary', '.min')

    def get_groups_and_computes_input(self, types, group_list,
            atom_group_prepend):
        groups_and_computes = []
        for type in types:
            groups_and_computes.append('group atom_type_'+type+' type '+
                                        type)
        for group in group_list:
            for type in types:
                groups_and_computes.append('\n#Atom '+group+' Type '
                                           + type+' Computes')
                groups_and_computes.append('group type_'+type+'_'+group+
                                '_atoms intersect atom_type_'+type+' '
                                + atom_group_prepend+'_'+group+'_atoms')
                groups_and_computes.append('compute type_'+type+'_'+group+
                                '_energy type_'+type+'_'+group+'_atoms reduce '
                                + 'sum c_per_atom')
                groups_and_computes.append('variable number_of_type_'+type+
                                '_'+group+'_atoms equal count(type_'+type+
                                '_'+group+'_atoms)')
                groups_and_computes.append('variable '+group+'_type_'+type+
                                '_energy_per_atom equal c_type_'+type+
                                '_'+group+'_energy/v_number_of_type_'+type+
                                '_'+group+'_atoms')
        return groups_and_computes

    def create_interface_input(self, box_bounds, name, potential_types, types,
            meam_files = None):
        setup = ge.Setup()
        standard_input = setup.get_standard_input('LAMMPS_Interface_Simulation')
        name_index = standard_input.index('#NAME')
        standard_input[name_index] = '#'+name
        if potential_types[0][0] == 'Meam/c LAMMPS':
            standard_input = self.set_potential_style_meam(
                standard_input, potential_types, meam_files)
            cut_off = '8'
        else:
            standard_input = self.set_potential_style(
                standard_input, potential_types)
            cut_off = self.get_cut_off(potential_types)
        standard_input = self.add_file_to_read(standard_input, 1, name)
        standard_input = self.change_simulation_box_size(standard_input,
                                                         box_bounds)
        standard_input = self.add_file_to_read(standard_input, 2, name)
        standard_input = self.define_interface_region(
                         standard_input, 'frozen', cut_off, 'out')
        standard_input = self.define_interface_region(
                         standard_input, 'interface', cut_off)
        groups_and_computes = self.get_groups_and_computes_input(
                              types, ['interface_1', 'interface_2'], 'unfrozen')
        atom_type_computes_index = standard_input.index('#Atom Type Computes')
        standard_input = (standard_input[:atom_type_computes_index+1] +
                          groups_and_computes +
                          standard_input[atom_type_computes_index+1:])
        standard_input = self.define_thermo_output_interface(
            standard_input, types)
        setup.create_input_file(standard_input, name+'/'+name, '.interface')

    def create_standard_lammps_input(self, name, unitcell_1, unitcell_2,
            potential_types, types, meam_files = None):
        self.create_single_cell_files_for_interface(
            name, unitcell_1, unitcell_2, potential_types)
        standard_input = ['\n\nclear\n\n']
        standard_input += self.create_input_for_single_LAMMPS_simulation(
            name+'_cell_1_only', 1, potential_types, types, meam_files)
        standard_input += ['\n\nclear\n\n']
        standard_input += self.create_input_for_single_LAMMPS_simulation(
            name+'_cell_2_only', 2, potential_types, types, meam_files)
        with open(name+'/'+name+'.interface', 'a+') as file:
            file.write('\n'.join(standard_input))

    def create_input_for_single_LAMMPS_simulation(self, name, unitcell_number,
            potential_types, types, meam_files = None):
        setup = ge.Setup()
        standard_input = setup.get_standard_input('LAMMPS_NPT_Simulation')
        name_index = standard_input.index('#NAME')
        standard_input[name_index] = '#'+name
        if potential_types[0][0] == 'Meam/c LAMMPS':
            stanard_input = self.set_potential_style_meam(
                standard_input, potential_types, meam_files)
        else:
            standard_input = self.set_potential_style(
                standard_input, potential_types)
        standard_input = self.add_single_file_to_read(
            standard_input, unitcell_number)
        groups_and_computes = self.get_single_groups_and_computes_input(types)
        atom_compute_index = standard_input.index('#Atom Type Computes')
        standard_input = (
            standard_input[:atom_compute_index+1]+groups_and_computes
            +standard_input[atom_compute_index+1:])
        standard_input = self.define_thermo_output_single(standard_input, types)
        dump_index = standard_input.index(
            'dump MiDump all custom 1000 #NAME.xyz id type xs ys zs fx fy fz &')
        standard_input[dump_index] = standard_input[dump_index].replace(
            '#NAME', name)
        return standard_input

    def create_single_cell_files_for_interface(self, name, unitcell_1,
            unitcell_2, potential_types):
        cell_repeat = self.get_required_supercell_repeats_for_adequete_atoms(
            unitcell_1, 8000)
        supercell_1 = ge.SuperCellExtended(
            unitcell_1, cell_repeat, cell_repeat, cell_repeat)
        cell_repeat = self.get_required_supercell_repeats_for_adequete_atoms(
            unitcell_2, 8000)
        supercell_2 = ge.SuperCellExtended(
            unitcell_2, cell_repeat, cell_repeat, cell_repeat)
        supercell_1.create_lammps_data_file(
            name+'/cell_1_only', potential_types)
        supercell_2.create_lammps_data_file(
            name+'/cell_2_only', potential_types)
        end_name = name.split('/')[-1]

    def get_single_groups_and_computes_input(self, types):
        groups_and_computes = []
        for type in types:
            groups_and_computes.append('group atom_type_'+type+' type '+
                                        type)
        for type in types:
            groups_and_computes.append('\n#Atom Type '+type+' Computes')
            groups_and_computes.append(
                'compute type_'+type+'_energy '+'atom_type_'+type+
                ' reduce sum c_per_atom')
            groups_and_computes.append(
                'variable number_of_type_'+type+'_atoms equal count(atom_'+
                'type_'+type+')')
            groups_and_computes.append(
                'variable atom_type_'+type+'_energy_per_atom equal '+
                'c_type_'+type+'_energy/v_number_of_type_'+type+'_atoms')
        return groups_and_computes

    def define_thermo_output_single(self, standard_input, types):
        thermo_index = standard_input.index('#Thermo output style')+1
        for type in types:
            standard_input[thermo_index] += (
                ' c_type_'+type+'_energy'+
                ' v_number_of_type_'+type+'_atoms '+
                'v_atom_type_'+type+'_energy_per_atom')
        return standard_input

    def get_required_supercell_repeats_for_adequete_atoms(self, unitcell,
            required_atoms):
        atom_number = unitcell.get_attributes()
        atom_number = [ atom for key, atom in atom_number.items() if
                        key[:4] == 'Atom']
        atom_number = len(atom_number)
        return int(round((required_atoms/atom_number)**(1/3)))

    def add_single_file_to_read(self, standard_input, cell_number):
        read_data_index = standard_input.index('read_data')
        standard_input[read_data_index] += ' cell_'+str(cell_number)+'_only.in'
        return standard_input

    def set_potential_style(self, standard_input, potential_types):
        cut_off = self.get_cut_off(potential_types)
        pair_style_index = standard_input.index('pair_style')
        standard_input[pair_style_index] += (
            ' '+potential_types[0][0].lower()+' '+cut_off)
        return standard_input

    def set_potential_style_meam(self, standard_input, potential_types,
            meam_files):
        pair_style_index = standard_input.index('pair_style')
        element_references = ' '.join(meam_files['Element_References'])
        standard_input[pair_style_index] += (
            ' '+potential_types[0][0].lower().split(' ')[0])
        pair_coeff_index = standard_input.index('#Pair_Coeff if MEAM')
        standard_input[pair_coeff_index] += (
            '\npair_coeff * * library.meam '+element_references+' '
            + meam_files['Library']+' '+element_references)
        return standard_input

    def get_cut_off(self, potential_types):
        setup = ge.Setup()
        cut_off = setup.get_potential_input(potential_types)
        cut_off = [float(atom[-1].split(' ')[-1]) for atom in cut_off]
        cut_off.sort()
        cut_off = str(cut_off[-1])
        return cut_off

    def change_simulation_box_size(self, standard_input, box_bounds):
        change_box_index = standard_input.index('change_box all')
        standard_input[change_box_index] += (
            ' x final '+str(box_bounds['X_Min'])+' '+str(box_bounds['X_Max'])+
            ' y final '+str(box_bounds['Y_Min'])+' '+str(box_bounds['Y_Max'])+
            ' z final '+str(box_bounds['Z_Min'])+' '+str(box_bounds['Z_Max'])+
            ' units box')
        return standard_input

    def add_file_to_read(self, standard_input, read_number, name):
        read_number = str(read_number)
        read_data_index = standard_input.index('read_data #'+read_number)
        standard_input[read_data_index] = (
                standard_input[read_data_index][:-2]+name
                + '_cell_'+read_number+'.in group interface_'+read_number
                + '_atoms')
        if int(read_number) > 1:
            standard_input[read_data_index] += ' add append'
        return standard_input

    def define_interface_region(self, standard_input, region_name,
            cut_off, side = 'in'):
        region_index = standard_input.index('region '+region_name
                                            + '_region block')
        if side == 'out':
            standard_input[region_index] += (' $(xlo+'+cut_off+') '+
                    '$(xhi-'+cut_off+') $(ylo+'+cut_off+') $(yhi-'+cut_off+') '+
                    '$(zlo+'+cut_off+') $(zhi-'+cut_off+') side out')
        elif side == 'in':
            standard_input[region_index] += (' $(xlo+'+cut_off+') '+
                    '$(xhi-'+cut_off+') $(ylo+'+cut_off+') $(yhi-'+cut_off+') '+
                    '$(zlo+'+cut_off+') $(zhi-'+cut_off+')')
        return standard_input

    def get_atom_types(self, atom_list):
        number_of_types = len(set([ atom[0] for atom in atom_list]))
        types = [ str(number) for number in range(1, number_of_types+1)]
        return types

    def define_thermo_output_interface(self, standard_input, types):
        thermo_index = standard_input.index(
            'v_number_of_unfrozen_interface_2_atoms '
            + 'v_interface_2_energy_per_atom')
        for type in types:
            standard_input[thermo_index] += (
                ' c_type_'+type+'_interface_1_energy'+
                ' v_number_of_type_'+type+'_interface_1_atoms '+
                'v_interface_1_type_'+type+'_energy_per_atom'+
                ' c_type_'+type+'_interface_2_energy '+
                'v_number_of_type_'+type+'_interface_2_atoms '+
                'v_interface_2_type_'+type+'_energy_per_atom')
        return standard_input

    def get_nearest_neighbours(self, atom, atom_list, neighbours):
        if len(atom) > 3:
            atom_list = [atom[1:] for atom in atom_list]
            atom = atom[1:]
        neighbour_tree = spatial.cKDTree(atom_list)
        nearest_neighbours = neighbour_tree.query(atom, k = neighbours)
        distance_list = list(nearest_neighbours[0])
        return distance_list



class AnalyseInterface():

    def get_excel_spreadsheet_of_model_output(self, output_file):
        excel_data = self.get_interface_output_data(output_file)
        work_book = xlwt.Workbook()
        sheet = work_book.add_sheet('Interface Model Output')
        y_counter = 2
        for key, run_data in excel_data.items():
            x_counter = 2
            sheet.write(y_counter, x_counter, key)
            y_counter += 2
            for header in run_data['Headers']:
                sheet.write(y_counter, x_counter, header)
                x_counter += 1
            y_counter += 1
            for data_line in run_data['Data']:
                x_counter = 2
                for data_point in data_line:
                    sheet.write(y_counter, x_counter, data_point)
                    x_counter += 1
                y_counter += 1
            y_counter += 1
        work_book.save(output_file.split('.')[0]+'.xls')

    def get_interface_output_data(self, output_file):
        with open(output_file, 'r') as file:
            output_data = file.read().split('\n')
        output_begin = [index for index in range(len(output_data)) if
                        output_data[index][:9] == 'Step Temp']
        output_end = [index for index in range(len(output_data)) if
                      output_data[index][:9] == 'Loop time']
        output_pairs = list(zip(output_begin,output_end))
        output_index_pairs = [{'Data_Begin': pair[0], 'Data_End': pair[1]}
                               for pair in output_pairs]
        interface_output_data = {}
        counter = 1
        for index_pair in output_index_pairs:
            interface_output_data['Run_'+str(counter)] = {
                'Headers': None, 'Data': None}
            headers = output_data[index_pair['Data_Begin']].split(' ')
            headers = [item for item in headers if item != '']
            interface_output_data['Run_'+str(counter)]['Headers'] = headers
            data = [
                [item for item in data_line.split(' ') if item != ''] for
                data_line in output_data[
                index_pair['Data_Begin']+1:index_pair['Data_End']]
                ]
            interface_output_data['Run_'+str(counter)]['Data'] = data
            counter += 1
        new_interface_output_data = {}
        counter = 1
        for run_number in range(1, len(interface_output_data)+1):
            if not run_number in [1,3,5,7]:
                new_interface_output_data['Run_'+str(counter)] = (
                    interface_output_data['Run_'+str(run_number)])
                counter += 1
        interface_output_data = new_interface_output_data
        for key, run in interface_output_data.items():
            for index in range(len(run['Data'])):
                run['Data'][index] = list(map(float, run['Data'][index]))
        region_index = [index for index in range(len(output_data)) if
            output_data[index][:29] == 'region interface_region block'][-1]
        region_sizes = output_data[region_index].split(' ')[3:]
        region_sizes = list(map(float, region_sizes))
        interface_output_data['Run_1']['Interface_X_Length'] = (
            region_sizes[1]-region_sizes[0])
        interface_output_data['Run_1']['Interface_Y_Length'] = (
            region_sizes[3]-region_sizes[2])
        interface_output_data['Run_1']['Interface_Area'] = (
            interface_output_data['Run_1']['Interface_X_Length']
            * interface_output_data['Run_1']['Interface_Y_Length'])
        return interface_output_data

    def get_data_averages(self, interface_output_data, lines_to_skip):
        for key, run in interface_output_data.items():
            if lines_to_skip > len(run['Data']):
                raise IndexError('The amount of lines you have chosen to skip '
                                 + ' is above the maximum line number in '+key
                                 + '.')
            averaged_data = run['Data'][lines_to_skip:]
            number_of_lines = len(averaged_data)
            averaged_data = np.array(averaged_data)
            averaged_data = averaged_data.transpose().tolist()
            averaged_data = [sum(data) for data in averaged_data]
            averaged_data = [data/number_of_lines for data in averaged_data]
            averaged_data = [round(data,9) for data in averaged_data]
            run['Averaged_Data'] = dict((zip(run['Headers'], averaged_data)))

    def get_interface_energy(self, interface_output_data):
        averaged_data_interface = interface_output_data['Run_2'][
                                  'Averaged_Data']
        averaged_data_cell_1 = interface_output_data['Run_3']['Averaged_Data']
        averaged_data_cell_2 = interface_output_data['Run_4']['Averaged_Data']
        atom_types = {key: item for key, item in averaged_data_cell_1.items()
                      if key[:16] == 'v_number_of_type'}
        type_data = {}
        for index in range(1, len(atom_types)+1):
            type_data['Type_'+str(index)] = {}
            type_data['Type_'+str(index)]['Atoms_In_Interface_1'] = (
                averaged_data_interface['v_number_of_type_'+str(index)
                + '_interface_1_atoms'])
            type_data['Type_'+str(index)]['Atoms_In_Interface_2'] = (
                averaged_data_interface['v_number_of_type_'+str(index)
                + '_interface_2_atoms'])
            type_data['Type_'+str(index)]['Energy/Atom_Cell_1'] = (
                averaged_data_cell_1['v_atom_type_'+str(index)
                + '_energy_per_atom'])
            type_data['Type_'+str(index)]['Energy/Atom_Cell_2'] = (
                averaged_data_cell_2['v_atom_type_'+str(index)
                + '_energy_per_atom'])
        expected_energy = 0
        for key, type in type_data.items():
            expected_energy += (
                (type['Atoms_In_Interface_1'] * type['Energy/Atom_Cell_1'])
                + (type['Atoms_In_Interface_2'] * type['Energy/Atom_Cell_2']))
        interface_output_data['Interface_Energy'] = (
            averaged_data_interface['c_interface_energy']-expected_energy)
        interface_output_data['Interface_Energy/Area'] = (
            interface_output_data['Interface_Energy']/
            interface_output_data['Run_1']['Interface_Area'])
        print('Interface Energy: '
              + str(interface_output_data['Interface_Energy'])+' (eV)')
        print('Interface Area: '
              + str(interface_output_data['Run_1']['Interface_Area'])
              + ' (Angstrom^2)')
        print('Interface Energy/Area: '
              + str(interface_output_data['Interface_Energy/Area'])
              + ' (eV/Angstrom^2)')
        print('Interface Size in X: '
              + str(interface_output_data['Run_1']['Interface_X_Length'])
              + ' (Angstrom)')
        print('Interface Size in Y: '
              + str(interface_output_data['Run_1']['Interface_Y_Length'])
              + ' (Angstrom)')

    def get_output_information(self, file):
        interface_output_data = self.get_interface_output_data(file)
        self.get_data_averages(interface_output_data, 3)
        self.get_interface_energy(interface_output_data)
        succint_interface_data = {
            key: data for key, data in interface_output_data.items() if
            key[:3] != 'Run'}
        for key, run in {
                key: data for key, data in interface_output_data.items() if
                key[:3] == 'Run'}.items():
            print(key)
            print(run['Averaged_Data'])
        print(succint_interface_data)

    def get_interface_output_data_temporary(self, output_file):
        with open(output_file, 'r') as file:
            output_data = file.read().split('\n')
        output_begin = [index for index in range(len(output_data)) if
                        output_data[index][:9] == 'Step Temp']
        output_end = [index for index in range(len(output_data)) if
                      output_data[index][:9] == 'Loop time']
        output_pairs = list(zip(output_begin,output_end))
        output_index_pairs = [{'Data_Begin': pair[0], 'Data_End': pair[1]}
                               for pair in output_pairs]
        interface_output_data = {}
        counter = 1
        for index_pair in output_index_pairs:
            interface_output_data['Run_'+str(counter)] = {
                'Headers': None, 'Data': None}
            headers = output_data[index_pair['Data_Begin']].split(' ')
            headers = [item for item in headers if item != '']
            interface_output_data['Run_'+str(counter)]['Headers'] = headers
            data = [
                [item for item in data_line.split(' ') if item != ''] for
                data_line in output_data[
                index_pair['Data_Begin']+1:index_pair['Data_End']]
                ]
            interface_output_data['Run_'+str(counter)]['Data'] = data
            counter += 1
        new_interface_output_data = {}
        counter = 1
        print([key for key in interface_output_data])
        for run_number in range(1, len(interface_output_data)+1):
            if not run_number in [1,3,5,6,8,9,11]:
                new_interface_output_data['Run_'+str(counter)] = (
                    interface_output_data['Run_'+str(run_number)])
                counter += 1
        interface_output_data = new_interface_output_data
        for key, run in interface_output_data.items():
            for index in range(len(run['Data'])):
                run['Data'][index] = list(map(float, run['Data'][index]))
        region_index = [index for index in range(len(output_data)) if
            output_data[index][:29] == 'region interface_region block'][-1]
        region_sizes = output_data[region_index].split(' ')[3:]
        region_sizes = list(map(float, region_sizes))
        interface_output_data['Run_1']['Interface_X_Length'] = (
            region_sizes[1]-region_sizes[0])
        interface_output_data['Run_1']['Interface_Y_Length'] = (
            region_sizes[3]-region_sizes[2])
        interface_output_data['Run_1']['Interface_Area'] = (
            interface_output_data['Run_1']['Interface_X_Length']
            * interface_output_data['Run_1']['Interface_Y_Length'])
        return interface_output_data

    def get_output_information_temporary(self, file):
        interface_output_data = self.get_interface_output_data_temporary(file)
        self.get_data_averages(interface_output_data, 3)
        self.get_interface_energy(interface_output_data)
        succint_interface_data = {
            key: data for key, data in interface_output_data.items() if
            key[:3] != 'Run'}
        for key, run in {
                key: data for key, data in interface_output_data.items() if
                key[:3] == 'Run'}.items():
            print(key)
            print(run['Averaged_Data'])
        print(succint_interface_data)












    def meaningless_function_to_extend_blank_space(self):
        pass
