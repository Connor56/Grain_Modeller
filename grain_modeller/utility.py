'''
Name:
    Utility
Description:
    Contains general use functions for common tasks performed within the
package.
'''
import numpy as np
import os


def get_best_normal(normals, centre, point):
    '''
    Take in an array of unit normal vectors, along with a centre, which
    represents the central point of an atom group, and a point which represents
    a single point from within that atom group. The normals represent planes.
    Assuming the shape is a convex hull, use the direction from the centre to
    the point in the plane to decide which normal direction best points out
    from the centre.

    normals: Numpy array of plane normals.
    centre: Numpy array of a central point of grouped atoms, for our purposes
        it represents the point of a direction vector.
    point: Numpy array of a point in the plane, the end point of the
        aforementioned direction vector.
    return: The best normal from the array as a vector in the form of a numpy
        array.
    '''
    direction = point-centre
    direction_mask = direction < 0 | np.isclose(direction, 0, atol=0.001)
    normal_masks = normals < 0 | np.isclose(normals, 0, atol=0.001)
    normal_mask = np.sum(normal_masks == direction_mask, axis=1)
    sorted_mask = np.flip(np.argsort(normal_mask))
    normal = normals[sorted_mask[0]]
    return normal


def order_difference(value1, value2, order=10):
    '''
    Calculates the difference in order between two values, gives it as a
    float. The default order is 10, but can be set to anything.
    '''
    order1 = np.log(value1)/np.log(order)
    order2 = np.log(value2)/np.log(order)
    difference = abs(order1 - order2)
    return difference


def read_atom_file(read_file):
    '''
    Takes in a LAMMPS data file and reads the atomic information out of it,
    returning the atomic positions in the form of a numpy array. Array only
    contains the atomic coordinates, no other information.

    file: LAMMPS data input file.
    return: Numpy array of atomic coordinates.
    '''
    with open(read_file, 'r') as file:
        data = file.read().split('\n')
        atoms = data.index('Atoms')
        masses = data.index('Masses')
        atom_data = data[atoms+2:masses-1]
        atom_data = [atom.split(' ')[2:] for atom in atom_data]
    return np.array(atom_data).astype(np.float64)


def get_standard_input(simulation_type):
    package_directory = self.get_package_directory()
    with open(package_directory+'Standard_Input.in', 'r') as simulation_file:
        simulation_data = simulation_file.read().split('\n')
        simulation_index = simulation_data.index('SIMULATION_TYPE: '+
                           simulation_type)+1
        simulation_end_index = simulation_data[simulation_index:].index(
                               'STANDARD_INPUT_END')+simulation_index
        standard_input = simulation_data[simulation_index:
                                         simulation_end_index]
    return standard_input


def create_input_file(file_input, name, file_type):
    with open(name+file_type, 'w') as gulp_file:
        file_input = '\n'.join(file_input)
        gulp_file.write(file_input)


def get_package_directory():
    current_directory = os.getcwd().split('/')
    try:
        package_directory_index = current_directory.index('grain_modeller')
        package_directory = ('../'*(len(current_directory)-
                            package_directory_index-1))
    except ValueError:
        package_directory = '../grain_modeller/'
    return package_directory


def get_element_masses(element_list):
    package_directory = get_package_directory()
    with open(package_directory+'grain_modeller/element_masses') as file:
        element_data = file.read().split('\n')
        element_data = element_data[:-1]
    updated_element_list = []
    for element in element_list:
        try:
            element_mass = [mass.split(' ')[1] for mass in element_data if
                            mass.split(' ')[0] == element+':'][0]
        except IndexError:
            raise ValueError('The element '+element+' does not have a '+
                             'mass listed in the element masses file. '+
                             'Please check the spelling of the element, '+
                             'or add its mass to the element masses File')
        updated_element_list.append(element+' '+element_mass)

    return updated_element_list


def get_potential_input(potential_types):
    package_directory = self.get_package_directory()
    with open(package_directory+'Potential_File.pot', 'r') as potential_file:
        potential_data = potential_file.read().split('\n')
        potential_input = []
        for potential in potential_types:
            potential_index = potential_data.index('POTENTIAL_TYPE: '+
                              potential[0])+1
            potential_end_index = potential_data[potential_index:].index(
                                  'POTENTIAL_LIST_END')+potential_index
            potential_slice = potential_data[potential_index:
                                             potential_end_index]
            try:
                potential_params = [ line.split(':')[1][2:-1] for line
                                    in potential_slice if line.split(':')[0]
                                    == potential[1]+' '+potential[2] or
                                    line.split(':')[0] == potential[2]+' '+
                                    potential[1]][0]
            except IndexError:
                raise ValueError('There is no potential for '+str(
                                 potential)+'. Please check you have the '+
                                 'right elements and potential type. If '+
                                 'you do then add the potential you need '+
                                 'to the Potential_File.pot')
            potential_params = potential_params.split(',')
            potential_append_list = [potential[0].lower(),potential[1],
                                     potential[2],
                                     ' '.join(potential_params)]
            potential_input.append(potential_append_list)
    return potential_input
