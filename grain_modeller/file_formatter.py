'''
Name:
    File Formatter
Description:
    Uses file templates and formatting values to create different file types,
    for example: LAMMPS data files, xyz files, and LAMMPS input scripts.
'''
import os
import re
import grain_creation as gc
import numpy as np
import utility
from supercell import SuperCell
import time


def format_file(formatting_object, template, directory='file_templates'):
    '''
    Creates a file based on the given template and a formatting object. File
    template has to be in the given directory (default is 'file_templates'),
    you can create your own templates by following the same syntax seen in the
    other templates. Formatting objects have to contain the data required by
    the file template, if the object is not a dictionary you will have to write
    your own helper function to format it into a dictionary of values.

    formatting_object: Python object that contains the information required to
        fill out a given template.
    template: Name of the template to search for in the given directory.
    directory: The directory to search for templates.
    '''
    template, variables = read_file(template, directory)
    formatting_object = check_formatting_object(formatting_object, variables)
    template = format_string(template, formatting_object)
    return template


def read_file(template, directory):
    '''
    Search for a file with the same name as template in the given directory.
    Read in the file and return the required variables, as well as the possible
    variables, along with the file body as a string.
    '''
    with open(directory+'/'+template+'.template', 'r') as file:
        data = file.read()
    required = [
        match.span() for match in re.finditer('REQUIRED_VARIABLES', data)]
    required = data[required[0][1]+1: required[1][0]]
    required = required.replace('\n', '').replace(' ', '').split(',')
    required = {index: None for index in required}
    possible = [
        match.span() for match in re.finditer('POSSIBLE_VARIABLES', data)]
    possible = data[possible[0][1]+1: possible[1][0]]
    possible = possible.replace('\n', '').replace(' ', '').split(',')
    possible = {index: None for index in possible}
    variables = {'Required': required, 'Possible': possible}
    template = [match.span() for match in re.finditer('TEMPLATE', data)]
    template = data[template[0][1]+1: template[1][0]]
    return (template, variables)


def check_formatting_object(formatting_object, variables):
    '''
    Formatting object type decides method for checking variables. For example a
    grain shape object (from grain_creation) is handled differently from a
    dictionary. However, all objects should produce a final dictionary of
    values which can be used to fill the given variable dictionary.
    '''
    if isinstance(formatting_object, gc.Grain):
        formatting_object = format_grain(formatting_object)
    if isinstance(formatting_object, SuperCell):
        formatting_object = format_supercell(formatting_object)

    for key in variables['Required']:
        try:
            variables['Required'][key] = formatting_object[key]
        except KeyError:
            raise KeyError(f'Formatting object does not have variable '
                           + f'{key}, check you are using the correct object '
                           + 'or that the file type is what you expect.')
    for key in variables['Possible']:
        try:
            variables['Possible'][key] = formatting_object[key]
        except KeyError:
            variables['Possible'][key] = None
    return variables


def format_grain(grain):
    '''
    Creates a dictionary of variables out of a grain object.
    '''
    supercell = grain.supercell
    formatting_object = format_supercell(grain.supercell)
    border = grain.border
    if not isinstance(border, list): border = [border]*6
    x_box_minimum = np.min(supercell.cartesian['coordinates'][:, 0])-border[0]
    x_box_maximum = np.max(supercell.cartesian['coordinates'][:, 0])+border[1]
    y_box_minimum = np.min(supercell.cartesian['coordinates'][:, 1])-border[2]
    y_box_maximum = np.max(supercell.cartesian['coordinates'][:, 1])+border[3]
    z_box_minimum = np.min(supercell.cartesian['coordinates'][:, 2])-border[4]
    z_box_maximum = np.max(supercell.cartesian['coordinates'][:, 2])+border[5]
    formatting_object['Name'] = grain.name
    formatting_object['X_Box_Minimum'] = x_box_minimum
    formatting_object['X_Box_Maximum'] = x_box_maximum
    formatting_object['Y_Box_Minimum'] = y_box_minimum
    formatting_object['Y_Box_Maximum'] = y_box_maximum
    formatting_object['Z_Box_Minimum'] = z_box_minimum
    formatting_object['Z_Box_Maximum'] = z_box_maximum
    return formatting_object


def array_to_string(array):
    '''
    Creates a string representation out of the atoms from a supercell
    structured array.
    '''
    array = np.hstack((array['element'][:, None], array['coordinates']))
    array = [' '.join(atom) for atom in array]
    array = '\n'.join(array)
    return array


def format_string(template, formatting_object):
    '''
    Replaces variable markers in the template string with the values given by
    the formatting object.
    '''
    for key, item in formatting_object['Required'].items():
        template = template.replace('--#'+key+'#--', str(item))
    for key, item in formatting_object['Possible'].items():
        if isinstance(item, type(None)):
            indexes = [match.span() for match
                       in re.finditer('\nPOSSIBLE:'+key, template)]
            if indexes == []:
                continue
            template = template[:indexes[0][0]] + template[indexes[1][1]:]
        else:
            template = template.replace('POSSIBLE:'+key+'\n', '')
            template = template.replace('\nPOSSIBLE:'+key, '')
            template = template.replace(f'--#{key}#--', str(item))
    return template


def format_supercell(supercell):
    '''
    Creates a dictionary of variables out of a supercell object.
    '''
    if supercell.cartesian is None: supercell.set_cartesian()
    number_of_atoms = supercell.fractional.shape[0]
    unique_atoms = np.unique(supercell.fractional['element'])
    number_of_atom_types = unique_atoms.shape[0]
    border = 0
    if not isinstance(border, list): border = [border]*6
    x_box_minimum = np.min(supercell.cartesian['coordinates'][:, 0])-border[0]
    x_box_maximum = np.max(supercell.cartesian['coordinates'][:, 0])+border[1]
    y_box_minimum = np.min(supercell.cartesian['coordinates'][:, 1])-border[2]
    y_box_maximum = np.max(supercell.cartesian['coordinates'][:, 1])+border[3]
    z_box_minimum = np.min(supercell.cartesian['coordinates'][:, 2])-border[4]
    z_box_maximum = np.max(supercell.cartesian['coordinates'][:, 2])+border[5]
    xy, xz, yz = 0, 0, 0
    fractional = array_to_string(supercell.fractional)
    cartesian = array_to_string(supercell.cartesian)
    masses = '\n'.join(utility.get_element_masses(unique_atoms.tolist()))
    masses_lammps = masses[:]
    cartesian_lammps = cartesian[:]
    for atom, count in zip(unique_atoms, list(range(1, len(unique_atoms)+1))):
        masses_lammps = masses_lammps.replace(atom, str(count))
        cartesian_lammps = cartesian_lammps.replace(atom, str(count))
    cartesian_lammps = cartesian_lammps.split('\n')
    atom_ids = map(str, list(range(1, len(cartesian_lammps)+1)))
    cartesian_lammps = '\n'.join([
        ' '.join(atom) for atom in list(zip(atom_ids, cartesian_lammps))])
    formatting_object = {
        'Name': 'Supercell', 'Number_of_Atoms': number_of_atoms,
        'Number_of_Atom_Types': number_of_atom_types,
        'X_Box_Minimum': x_box_minimum, 'X_Box_Maximum': x_box_maximum,
        'Y_Box_Minimum': y_box_minimum, 'Y_Box_Maximum': y_box_maximum,
        'Z_Box_Minimum': z_box_minimum, 'Z_Box_Maximum': z_box_maximum,
        'XY': xy, 'XZ': xz, 'YZ': yz, 'Atoms_Fractional': fractional,
        'Atoms_Cartesian': cartesian,
        'Atoms_Cartesian_LAMMPS': cartesian_lammps, 'Masses': masses,
        'Masses_LAMMPS': masses_lammps}
    return formatting_object
