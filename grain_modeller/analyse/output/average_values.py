'''
Name:
    Average Values
Description:
    Contains functions that find the average position of each atom contained
    within a molecular dynamics output file. This is to produce averaged
    representations of grains for magnetic modelling.
'''

import numpy as np
import time

STYLES = {'Lammps_Grain_Positions':
          {'Split_Start': 'ITEM: ATOMS', 'Split_Stop': 'ITEM: TIMESTEP',
           'Columns': [0, 1, 2, 3, 4]}}

def get_file_data(file, style):
    '''
    Reads a molecular dynamics output file using the style as a guide for how
    to read it. For example you may want columns 1, 2, 5, 6 etc.

    file: String of file name to be read.
    style: File output type, defines how to read the file.
    '''
    if isinstance(style, str): style = STYLES[style]

    with open(file, 'r') as out_file:
        out_data = out_file.read().split('\n')
        start_indexes = [
            index+1 for index in range(len(out_data)) if style['Split_Start']
            in out_data[index]]
        stop_indexes = [
            index for index in range(len(out_data)) if style['Split_Stop'] in
            out_data[index]]
        stop_indexes = [
            index for index in stop_indexes if index > min(start_indexes)]
        indexes = list(zip(start_indexes, stop_indexes))
        atom_values = []
        for index in indexes:
            step_data = out_data[index[0]: index[1]]
            step_data = [step.split(' ') for step in step_data]
            step_data = [
                [step[index] for index in range(len(step)) if index in
                 style['Columns']] for step in step_data]
            atom_values.append(step_data)
    return atom_values


def get_average_positions(file, style, step_start, step_end=None):
    '''
    Averages the positions of all the atoms in a grain.

    file: String of file name to be read.
    style: File output type, defines how to read file.
    step_start: The step number to start counting from.
    step_end: The step number to stop counting from.
    '''
    position_data = get_file_data(file, style)

    if step_start < 0:
        step_start = len(position_data)+step_start
    if step_start < 0:
        raise ValueError("Starting step is too negative, step: % doesn't exist"
                         % step_start)
    if step_end is not None and step_end < 0:
        step_end = len(position_data)+step_end
    if step_end is not None and step_end < 0:
        raise ValueError("Ending step is too negative, step: % doesn't exist"
                         % step_start)

    if step_end is None:
        position_data = position_data[step_start:]
    elif step_start < step_end:
        position_data = position_data[step_start: step_end]
    elif step_start > step_end:
        raise ValueError(
            'step_end position comes before the step_start position')

    position_data = np.array(position_data).astype(float)
    num_steps = position_data.shape[0]
    position_data = position_data.reshape(-1, 5)
    position_data = position_data[np.argsort(position_data[:, 0])]
    position_data = position_data.reshape(-1, num_steps, 5)
    average_positions = {
        int(value[0, 0]): np.mean(value[:, 1:], 0) for value in position_data}
    return average_positions


def create_file_from_averaged_data_lammps(average_data, name, box_padding=0):
    '''
    Using the averaged data create a lammps data input file.

    average_data: Averaged per atom data for the input file.
    name: Name of file, and title of data input in string format.
    box_padding: Amount of empty space to add between the most extreme atom in
        all primary directions and the edge of the simulation box, units are
        whatever the base unit of the simulations are (generally Angstrom).
    '''
    atoms = [[key]+[int(value[0])]+value[1:].tolist() for key, value
             in average_data.items()]
    atom_array = np.array(atoms)
    xlo = atom_array[:, 2].min()-box_padding
    xhi = atom_array[:, 2].max()+box_padding
    ylo = atom_array[:, 3].min()-box_padding
    yhi = atom_array[:, 3].max()+box_padding
    zlo = atom_array[:, 4].min()-box_padding
    zhi = atom_array[:, 4].max()+box_padding
    input_list = [
        name, '', '%s atoms' % len(atoms),
        '%s atom types' % len(np.unique(atom_array[:, 1])),
        '%s %s xlo xhi' % (xlo, xhi), '%s %s ylo yhi' % (ylo, yhi),
        '%s %s zlo zhi' % (zlo, zhi), '0 0 0 xy xz yz', '', 'Atoms', '']
    input_list = input_list+[' '.join(list(map(str, atom))) for atom in atoms]
    with open(name+'.in', 'w') as file:
        file.write('\n'.join(input_list))
