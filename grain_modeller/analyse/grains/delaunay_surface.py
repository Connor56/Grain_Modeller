'''
Name:
    Delaunay surface
Description:
    Uses the delaunay3d method in pyvista to find the surface of a grain.
'''

import numpy as np
import pyvista as pv
from analyse.grains import surface_base as sb


def analyse(points, angle=0.01, alpha=0):
    '''
    Defines a surface object from a point set. Surface object contains all the
    surface simplexes along with the total area, total enclosed volume, and
    the area grouped by simplex normal.

    angle: Maximum allowed angle between face normals (given in radians)
    alpha: Maximum allowed distance between simplex points, alpha in alpha
        shapes. 0 gives the default behaviour of the delaunay algorithm.
    '''
    print("Warning: When interpreting the results of the surface analysis be "
          "careful to remember surface normals are in cartesian coordinates. "
          "This means it's unlikely surface normals are directly related to "
          "the face normals of your crystal structure. To transform these "
          "normals into your crystal structure normals use the crystal "
          "structures inverse vector space.")
    shell = get_shell(points, alpha)
    surface = sb.get_surface(shell)
    surface.grouped_simplexes = sb.group_simplexes(surface.simplexes, angle)
    surface.grouped_area = sb.get_areas(surface.grouped_simplexes)
    return surface


def get_shell(atom_array, alpha=0, shift=True):
    '''
    Uses pyvista to tetrahedralize the input atom_array and separate out its
    surface shell from the rest of the shape. Returns a PolyData VTK object
    using the Pyvista package.

    alpha: Maximum allowed distance between simplex points, alpha in alpha
        shapes. 0 gives the default behaviour of the delaunay algorithm.
    shift: If True, shifts the atom_array to the origin - prevents an issue
        with inaccurate volume.
    '''
    if shift:
        shifted_atom_array = atom_array - np.min(atom_array, axis=0)
    else:
        shifted_atom_array = atom_array
    cloud = pv.PolyData(shifted_atom_array)
    volume = cloud.delaunay_3d(alpha=alpha)
    shell = volume.extract_geometry()
    return shell
