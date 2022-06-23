'''
Name:
    Cartesian Edits
Description:
    Editing functions for atoms outside of supercell. These are required when
    reading in and manipulating data not created within the program.
'''

import numpy as np
import warnings


def box_select(points, box, out=False):
    '''
    Returns a boolean array representing the points that fall inside a box
    (out=False), or outside a box (out=True).

    atoms: Numpy array of point coordinates in 3D.
    box: Numpy array, list, or tuple in the following form:
                            [[x_min, x_max],
                             [y_min, y_max],
                             [z_min, z_max]]
        These are used as the conditions of a boolean array.
    out: Boolean, False selects atoms within the box, True selects atoms
        outside the box.
    '''
    box = np.array(box)
    if box.shape != (3, 2):
        raise ValueError(
            f"Box dimesions are: {box.shape}, the required dimensions are "
            "(3, 2). Please read the box_select description for more details.")
    selected = np.all((points > box[:, 0]) & (points < box[:, 1]), axis=1)
    if np.sum(selected) == 0:
        warnings.warn(
            f"No atoms within box:\n{box}\nCheck box dimensions if this was "
            "not deliberate.")
    if out:
        selected = np.invert(selected)
    return selected
