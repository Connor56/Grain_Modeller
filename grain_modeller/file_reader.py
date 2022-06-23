'''
Name:
    File Reader
Description:
    Reads in information from output or data files that can be placed into an
    atomic format.
'''
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
from atom import Atom
from unitcell import UnitCell
from supercell import SuperCell


@dataclass
class StepData():
    time_step: int
    number_of_atoms: int
    box_data: pd.DataFrame
    atom_data: pd.DataFrame


@dataclass
class FileData():
    name: str
    type: str
    steps: tuple[StepData]


def read_file(file_name):
    '''
    Establishes the file type, and redirects it to a helper function.
    '''
    if file_name[-4] == '.xyz':
        data = xyz_file(file_name)
        return data
    if file_name[-4] == '.cif':
        data = cif_file(file_name)
        return data
    if file_name[-3] == '.in':
        data = lammps_file(file_name)
        return data


def xyz_file(file_name):
    '''
    Gets atomic information from the .xyz file format, specifically the data
    from LAMMPS output. Stores data for each output timestep in the file
    '''
    with open(file_name, 'r') as f:
        data = f.read().split('\n')
    timesteps = np.array(index_matches(data, 'TIMESTEP'))+1
    number_of_atoms = np.array(index_matches(data, 'NUMBER OF ATOMS'))+1
    box_data = np.array(index_matches(data, 'BOX BOUNDS'))+1
    atom_data = np.array(index_matches(data, 'ITEM: ATOMS'))
    steps = np.vstack((timesteps, number_of_atoms, box_data, atom_data)).T
    steps_data = []
    for step in steps:
        timestep = int(data[step[0]])
        number_of_atoms = int(data[step[1]])
        box_data = np.array([list(map(float, line.split(' ')))
                             for line in data[step[2]:step[2]+3]])
        box_data = pd.DataFrame({'Minimum': box_data[:, 0],
                                 'Maximum': box_data[:, 1]},
                                index='x y z'.split(' '))
        atom_headers = data[step[3]].split('ITEM: ATOMS')[1].split(' ')
        atom_headers = [header for header in atom_headers if header != '']
        atom_data = np.array([[i for i in line.split(' ') if i != ''] for line
                              in data[step[3]+1: step[3]+number_of_atoms+1]])
        atom_data = atom_data.astype(float)
        atom_data = pd.DataFrame({atom_headers[i]: atom_data[:, i] for i
                                  in range(len(atom_headers))})
        step_data = StepData(timestep, number_of_atoms, box_data, atom_data)
        steps_data.append(step_data)
    return FileData(file_name.replace('.xyz', ''), '.xyz', tuple(steps_data))


def index_matches(string_list, regex):
    '''
    Finds the indexes of all the strings in a string list which match a given
    regular expression.
    '''
    pattern = re.compile(regex)
    indexes = [index for index in range(len(string_list))
               if len(pattern.findall(string_list[index])) > 0]
    if indexes == []: raise ValueError(f"{regex} isn't present in list.")
    return indexes


def cif_file(file_name):
    '''
    Takes the data from a cif file and turns it into a supercell. Returns this
    supercell.
    '''
    with open(file_name, 'r') as f:
        cif_data = f.read()


def lammps_file(file_name):
    '''
    Takes the atomic information out of lammps data file and returns it as a
    numpy array.
    '''
    atom = (r"(\d+)[\s\t]+(\d)[\s\t]+([-\d\.]+)[\s\t]+"
            r"([-\d\.]+)[\s\t]+([-\d\.]+)")
    atom_pattern = re.compile(atom)
    with open(file_name, 'r') as f:
        data = f.read()
    atoms = atom_pattern.findall(data)
    atoms = np.array(atoms).astype(float)
    atoms = atoms[:, 1:]
    return atoms

















if True:
    pass
