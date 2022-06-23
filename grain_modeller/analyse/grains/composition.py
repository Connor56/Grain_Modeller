'''
Name:
    Composition
Description:
    Analyses the element composition of grains.
'''

import numpy as np


def is_identical(style, files):
    '''
    Tests if composition of different files is the same. Style defines the type
    of all the files to search. Every file should be the same type.
    '''
    compositions = []
    if style == 'LAMMPS':
        for file in files:
            with open(file, 'r') as f:
                data = f.read().split('\n')
            atom_start = data.index('Atoms')+2
            atom_end = data.index('Masses')-1
            atoms = data[atom_start: atom_end]
            atoms = list(map(int, [atom.split(' ')[1] for atom in atoms]))
            atoms = np.array(atoms)
            composition = [
                np.sum(atoms == unique) for unique in np.unique(atoms)]
            compositions.append(composition)
    return True if np.unique(compositions, axis=0).shape[0] == 1 else False
