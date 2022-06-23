'''
Name:
    Reconstruct Surface
Description:
    Uses the reconstruct_surface method from pyvista to find the surface of a
    grain.
'''

import numpy as np
import pyvista as pv
import linear_algebra as linalg
import crystallography
from analyse.grains import surface_base as sb


def analyse(points, box=None, neighbours=12, angle=0.01, show_surface=False):
    '''
    Defines a surface object from a point set. Similar to the analyse_surface
    function above, extracts surface simplexes from a point set getting area,
    volume and area gouped by normal. Predefines surface atoms, and uses a
    different method for suface construction.

    points: Numpy array of points in 3D.
    angle: Grouping angle for plane normals. (minimum allowed angle for
        grouping)
    box: Numpy array, list, or tuple of rows defining a the limits of a 3D box
        given by the following form:
                            [[x_min, x_max],
                             [y_min, y_max],
                             [z_min, z_mac]]
    '''
    print("Warning: When interpreting the results of the surface analysis be "
          "careful to remember surface normals are in cartesian coordinates. "
          "This means it's unlikely surface normals are directly related to "
          "the face normals of your crystal structure. To transform these "
          "normals into your crystal structure normals use the crystal "
          "structures inverse vector space.")
    surface_points = surface_atoms(points, box=box, neighbours=neighbours)
    surface_points = pv.wrap(surface_points)
    shell = surface_points.reconstruct_surface()
    if show_surface:
        shell.plot()
    surface = sb.get_surface(shell)
    surface.grouped_simplexes = sb.group_simplexes(surface.simplexes, angle)
    surface.grouped_area = sb.get_areas(surface.grouped_simplexes)
    return surface


def surface_atoms(atoms, box, neighbours=12):
    '''
    Finds the surface atoms by comparing the distance symmetries of bulk
    atoms, contained within the defined box, to all atoms. Atoms that diverge
    from bulk symmetry are surface atoms.

    atoms: Numpy array of atom coordinates (typically in cartesian).
    box: Numpy array, list, or tuple of rows defining the limits of a 3D box
        given by the following form:
                            [[x_min, x_max],
                             [y_min, y_max],
                             [z_min, z_mac]]
    neighbours: Number of neighbours to search for when checking distance
        symmetries.

    Unfortunately, this method does not work for all structures. If you get an
    unexpected result, put the returned surface atoms into a visualiser.
    '''
    bulk_symmetries = crystallography.distance_symmetries(
        atoms, neighbours, box)
    all_symmetries = crystallography.neighbour_distances(atoms, neighbours)
    surface_mask = linalg.match_rows(all_symmetries, bulk_symmetries)
    surface_mask = np.invert(surface_mask)
    return atoms[surface_mask]
