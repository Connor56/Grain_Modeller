'''
Name:
    Testing Tools.
Description:
    This module contains functions that help test code when it is not
    performing correctly. Functions added to this file generally have been
    used previously to test multiple times, to the extent it was worth
    keeping them somewhere to use when required.
'''
import os
import utility


def xyz_output(file_name, atom_list):
    '''
    Takes in a filename and an atom list and produces a simple xyz file
    from this input. All the atoms will be Fe atoms, as within this test
    suite all we are interested in is atomic positions.

    file_name: name of file to be created in form of string.
    atom_list: list of the atoms that make up the .xyz file.
    '''
    if len(atom_list[0]) == 3:
        atoms = [' '.join(['Fe']+list(map(str, atom))) for atom in atom_list]
        print_list = [str(len(atoms)), '']+atoms
    elif len(atom_list[0]) == 4:
        atoms = [' '.join(map(str, atom.tolist())) for atom in atom_list]
        print_list = [str(len(atoms)), '']+atoms
    elif len(atom_list[0]) == 2:
        atoms = list(zip(
            atom_list['element'].tolist(), atom_list['coordinates'].tolist()))
        atoms = [f'{atom[0]} {atom[1][0]} {atom[1][1]} {atom[1][2]}' for atom
                 in atoms]
        print_list = [str(len(atoms)), '']+atoms

    utility.create_input_file(print_list, file_name, '.xyz')


def plane_folder_creator(folder_number, planes):
    '''
    Takes in a plane folder number, along with a plane dictionary object.
    Produces the required folder and then creates a series of .xyz files
    of the planes in the dictionary.

    folder: integer to name the plane folder with.
    planes: dictionary of the planes to make .xyz files from, includes the
        atoms that make up the plane.
    '''
    folder_number = str(folder_number)
    try:
        os.mkdir('./planes_'+folder_number)
    except FileExistsError:
        pass
    for key, plane in planes.items():
        name = 'planes_'+folder_number+'/'+key
        xyz_file_output(name, plane['Plane_Atoms'].tolist())
