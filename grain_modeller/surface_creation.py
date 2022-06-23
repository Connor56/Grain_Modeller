'''
Name:
    Surface Creation
Description:
    Contains functions for finding surface energies. Works with a SuperCell to
    create the desired surfaces. It may be the case that some of the functions
    within this module would fit better inside the transforms.py module.
    Particularly the supercell reorientation one for example, which could be
    generalised fairly easily.
'''

import numpy as np
import copy
import transforms
import linear_algebra as linalg
from dataclasses import dataclass
import pandas as pd
import edits
import grain_creation as gc
import testing_tools
import crystallography


@dataclass
class Slab():
    '''
    Represents a cuboid slab, with a volume, points, and a point density.
    '''
    volume: float
    points: np.ndarray
    number_of_points: int
    z_min: float
    z_max: float
    point_density: float = None


def reorientate_supercell(supercell, plane):
    '''
    Reorientates the given supercell so that the given plane has its normal
    aligned along the cartesian z direction. The given plane is in the hkl
    miller indices of the supercell.
    '''
    vector_space = copy.deepcopy(supercell.vector_space)
    intercepts = crystallography.miller_to_intercepts(plane)
    vector = linalg.plane_normal(vector_space, intercepts)
    rotation_vector = get_rotation_vector(vector, np.array([0, 0, 1]))
    vector = linalg.normalise(vector)
    angle = linalg.angle(vector, np.array([0, 0, 1]))
    rotation_vector = rotation_vector/np.linalg.norm(rotation_vector)
    rotation_matrix = (
        linalg.rodrigues_rotation_matrix(angle, rotation_vector))
    if not np.isclose(rotation_matrix @ vector, np.array([0, 0, 1])).all():
        rotation_matrix = (
            linalg.vector_rotation_matrix(-angle, rotation_vector))
    transforms.rotate(supercell, rotation_matrix)


def get_rotation_vector(vector1, vector2):
    '''
    Using the cross product of two vectors, find and return the unit vector
    normal around which to rotate to move vector1 into the direction of vector2
    using a single rotation step.
    '''
    rotation_vector = np.cross(vector1, vector2)
    rotation_vector = linalg.normalise(rotation_vector)
    return rotation_vector


def cut_slab(supercell, thickness, minimum=0):
    '''
    Cut a slab from the given supercell with the given thickness if possible.
    This function takes a supercell under the assumption that the slab is
    orientated so that the plane of interest has its normal in the [001]
    direction. The function uses the mini_slab_thickness to divide up the
    supercell volume into mini slabs. The maximum and minimum extent of each
    slab can be used to decide how the supercell should be cut.
    The end result of this function is a cuboid slab of the desired thickness,
    if the desired thickness can't be achieved then an error is returned.

    supercell: SuperCell data Object.
    thickness: Desired slab thickness given in Angstrom.
    slab_thickness: Thickness of the mini slabs that divide up the supercell
        along the Z direction. Should be chosen so it's not possible for a slab
        to have no atoms in it.
    minimum: Is the minimum value the dimensions in x and y must change when
        finding the best dimensions to use for the slab.
    '''
    supercell.set_cartesian()
    points = supercell.cartesian['coordinates'].copy()
    working_volume = select_working_volume(points, thickness)
    points = working_volume.points
    xy_dimensions = get_xy_dimensions(points, minimum)
    z_dimensions = np.array([working_volume.z_min, working_volume.z_max])
    slab_limits = np.vstack((xy_dimensions.values, z_dimensions)).flatten()
    points = [[slab_limits[0], 0, 0], [slab_limits[1], 0, 0],
              [0, slab_limits[2], 0], [0, slab_limits[3], 0],
              [0, 0, slab_limits[4]], [0, 0, slab_limits[5]]]
    normals = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1],
               [0, 0, 1]]
    cut_list = []
    for index in range(6):
        cut_list.append(edits.Cut('cp', points[index], normals[index]))
    gc.cut_grain(supercell, cut_list)
    return supercell


def select_working_volume(points, thickness):
    '''
    Given a set of points, finds the best volume segment to begin the slab
    optimisation process with. If the deisred thickness cannot be achieved
    throws an error. Returns a Slab dataclass containing all the information
    pertaining to this slab volume.
    '''
    min_limits = np.min(points, axis=0)
    max_limits = np.max(points, axis=0)
    if thickness > (max_limits[2] - min_limits[2]):
        raise ValueError(
            f"Thickness: {thickness}, is too large. Try a smaller thickness "
            "or a larger supercell.")
    z_points = np.arange(min_limits[2], max_limits[2]-thickness+1, 1)
    slabs = split_into_slabs(points, thickness, z_points=z_points)
    slabs.sort(key=lambda x: x.point_density)
    return slabs[-1]


def split_into_slabs(points, thickness, step=1, z_points=None):
    '''
    Splits a structure into slabs along the z axis, using the list of z_points
    as the z minimum, and the thickness as the amount to add to that minimum to
    get the maximum z height of the slab. The points are what make up the
    structure to be split into slabs. Returns a list of slab data objects.
    '''
    min_limits = np.min(points, axis=0)
    max_limits = np.max(points, axis=0)
    try:
        z_points[0]
    except TypeError:
        z_points = np.arange(min_limits[2], max_limits[2]-thickness+step, step)
    slabs = []
    for point in z_points:
        slab_points = points.copy()
        slab_points = slab_points[slab_points[:, 2] >= point]
        slab_points = slab_points[slab_points[:, 2] < point+thickness]
        x_side, y_side = (max_limits - min_limits)[:2]
        volume = x_side*y_side*thickness
        point_density = get_point_density(slab_points.shape[0], volume)
        slab = Slab(volume, slab_points, slab_points.shape[0], point,
                    point+thickness, point_density)
        slabs.append(slab)
    return slabs


def get_point_density(points, volume):
    '''
    Gets the point density of volume enclosing N number of points.
    '''
    return points/volume


def get_xy_dimensions(points, minimum=0):
    '''
    Finds the xy dimensions that are required to have a cuboid slab. Starting
    from the centre of the slab cuts wider and wider whilst maintaining close
    to maximum point density. The minimum value represents the minimum step
    in dimension size used for the change dimension function. For some
    structures this is required to prevent too many minor shifts.
    '''
    max_values = np.max(points, axis=0)
    min_values = np.min(points, axis=0)
    dimensions = [min_values[0], max_values[0],
                  min_values[1], max_values[1]]
    completed = np.array([False, False, False, False])
    size_record = []
    count = 0
    while not np.all(completed):
        index = count % 4
        count += 1
        print(dimensions)
        if dimensions[0] >= dimensions[1]:
            break
        if dimensions[2] >= dimensions[3]:
            break
        current_points = points.copy()
        current_points = current_points[
            (current_points[:, 0] >= dimensions[0]) &
            (current_points[:, 0] < dimensions[1]) &
            (current_points[:, 1] >= dimensions[2]) &
            (current_points[:, 1] < dimensions[3])]
        if current_points.size == 0:
            break
        volume = ((dimensions[1]-dimensions[0])*(dimensions[3]-dimensions[2])*
                  (max_values[2] - min_values[2]))
        point_density = get_point_density(current_points.shape[0], volume)
        record = {'Dimensions': dimensions[:], 'Volume': volume,
                  'Points': current_points.shape[0],
                  'Point_Density': point_density}
        size_record.append(record)
        dimensions = change_dimension(dimensions, points, index, minimum)
    point_densities = np.array([x['Point_Density'] for x in size_record])
    point_densities = pd.DataFrame(point_densities)
    if point_densities.shape[0] > 64:
        roll = int(np.log(point_densities.shape[0]))*4
    else:
        roll = 4
    point_densities['averages'] = point_densities.rolling(roll).mean()
    point_densities['differences'] = point_densities['averages'].diff()
    point_densities['average_difference'] = (
        point_densities['differences'].rolling(roll).mean())
    indexes = np.around(point_densities['average_difference'], 10) <= 0
    indexes = indexes.index[indexes.values]
    xy_dimensions = np.array(
        size_record[indexes[0]]['Dimensions']).reshape([2, 2])
    xy_dimensions = pd.DataFrame(
        xy_dimensions, index=['x', 'y'], columns=['lower', 'upper'])
    return xy_dimensions


def best_minimum(points, minimum):
    '''
    Finds the best minimum distance change in the slabs to make sure that one
    side of the slab doesn't decrease at a much faster rate than the other.
    If the minimum chosen is smaller than the spacing in either direction the
    minimum will become the average spacing instead.
    '''


def change_dimension(dimensions, points, index, minimum=0):
    '''
    Takes in a set of points, and a set of dimensions in the x, y directions
    which define the desired limit of those points. The index defines the
    particular limit we're interested in by the following:
        index: 0, dimension: x minimum,
        index: 1, dimension: x maximum,
        index: 2, dimension: y minimum,
        index: 3, dimension: y maximum
    Selects the next smallest point for the desired dimension. For example if
    the current dimension gives x minimum of 5, finds the value of all points
    in x and selects the next smallest value, say 6, 7, or 5.5 etc. Next
    smallest value meaning a higher value than the current one. The minimum
    variable keeps the dimensions changing significantly enough that dimension
    shifts for a complicated structure isn't slowed down to the point that
    calculating the slab dimensions becomes prohibitive. Can be set if needs
    be.
    '''
    if index == 0:
        values = points[:, 0]
        values = values[values > dimensions[index]+minimum]
        dimensions[index] = np.min(values)
    elif index == 1:
        values = points[:, 0]
        values = values[values < dimensions[index]-minimum]
        dimensions[index] = np.max(values)
    elif index == 2:
        values = points[:, 1]
        values = values[values > dimensions[index]+minimum]
        dimensions[index] = np.min(values)
    elif index == 3:
        values = points[:, 1]
        values = values[values < dimensions[index]-minimum]
        dimensions[index] = np.max(values)
    return dimensions
