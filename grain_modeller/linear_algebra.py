'''
Name:
    Linear Algebra
Description:
    Contains matrix, and vector functions used in other modules.
'''
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R


def rotation_matrix(axis, angle):
    '''
    Create the 3D rotation matrix which rotates around the given axis.
    '''
    if axis == 'x':
        matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(angle), -np.sin(angle)],
             [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        matrix = np.array(
            [[np.cos(angle), 0, np.sin(angle)],
             [0, 1, 0],
             [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle), np.cos(angle), 0],
             [0, 0, 1]])
    return matrix


def angle(vector1, vector2):
    '''
    Minimum angle between two vectors: vector1 and vector2. Vectors should be
    input as numpy arrays.
    '''
    angle = np.arccos((np.dot(vector1, vector2))/(norm(vector1)*norm(vector2)))
    return angle


def vector_rotation_matrix(angle, rotation_vector):
    '''
    Creates a rotation matrix that rotates by an angle around a given rotation
    vector. The given vector must be a unitvector, and the angle must be in
    radians. The multiple of the unitvector normal decides the angle through
    which to turn. Rotation follows the right hand rule.
    '''
    return R.from_rotvec(angle*rotation_vector).as_matrix()


def rodrigues_rotation_matrix(angle, rotation_vector):
    '''
    Uses Rodrigues' formula for a rotation matrix, given separately here:
    1. https://mathworld.wolfram.com/RodriguesRotationFormula.html
    2. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    This formula produces a rotation matrix from a unit vector around which you
    wish to rotate, and the angle by which you wish to rotate. Therefore, you
    should input a unit vector as a rotation vector. If the rotation vector
    the function will throw a value error.
    '''
    if np.linalg.norm(rotation_vector) != 1:
        raise ValueError(f"Rotation Vector: {rotation_vector} is not a unit "
                         "vector. Please input unit vector to use function")
    K_matrix = np.array([[0, -rotation_vector[2], rotation_vector[1]],
                         [rotation_vector[2], 0, -rotation_vector[0]],
                         [-rotation_vector[1], rotation_vector[0], 0]])
    rotation_matrix = np.identity(3)
    rotation_matrix += np.sin(angle)*K_matrix
    rotation_matrix += (1-np.cos(angle))*(K_matrix @ K_matrix)
    return rotation_matrix


def normalise(array):
    '''
    Normalises all the vectors in a numpy array. Treats the rows of the array
    as the vectors to normalise. Can normalise single vectors also.
    '''
    if len(array.shape) == 1:
        normal_lengths = np.linalg.norm(array)
    else:
        normal_lengths = np.linalg.norm(array, axis=1)
        normal_lengths = normal_lengths[:, None]
    array = array/normal_lengths
    return array


def plane_normal(vector_space, intercepts):
    '''
    Using a given vector space and a series of intercepts along the vector
    space directions, get the plane normal within cartesian space. Vector space
    is in 3D, returns a unit normal moving away from the origin.

    This is useful for getting the absolute cartesian normal of a plane defined
    in a none orthonormal vector space. None orthonormal vector spaces commonly
    occur in crystals.
    '''
    intercepts = np.array(intercepts)
    points = plane_points(vector_space, intercepts)
    plane_vector_1 = points[1] - points[0]
    plane_vector_2 = points[2] - points[0]
    normal = np.cross(plane_vector_1, plane_vector_2)
    unit_normal = normal/np.linalg.norm(normal)
    distance_1 = np.linalg.norm(points[0] - unit_normal*0.1)
    distance_2 = np.linalg.norm(points[0] + unit_normal*0.1)
    if distance_1 > distance_2:
        unit_normal = -1*unit_normal
    return unit_normal


def plane_points(vector_space, intercepts):
    '''
    Uses a vector space and the intercepts of a plane to get three points in
    the plane which can be used for generating a plane normal. Vector space
    must be in 3D.
    '''
    intercepts = np.array(intercepts)
    if np.sum(np.array(intercepts) == 0) > 1:
        raise ValueError(
            f"Intercepts: {intercepts}, has too many intercepts at zero, only "
            "one is possible, otherwise points coincide and no plane can be "
            "defined. Make sure your intercepts are defined correctly.")
    if np.all(intercepts == np.inf):
        raise ValueError(
            f"Intercepts: {intercepts}, cannot all be infinite, a maximum of "
            "two is permissable when defining a plane. Make sure your "
            "intercepts are defined correctly.")
    # Find the list index of the first intercept value which is not infinite.
    none_infinite_intercepts = np.argwhere(intercepts != np.inf)
    not_infinite_index = none_infinite_intercepts[0]
    plane_points = []
    # Create three points on the plane.
    for i in range(3):
        plane_point = np.array([0., 0., 0.])
        if intercepts[i] == np.inf:
            # If an intercept is infinity, select a plane point above a
            # different intercept by the relevant vector.
            plane_point[i] = 1
            plane_point[not_infinite_index] = intercepts[not_infinite_index]
        else:
            plane_point[i] = intercepts[i]
        plane_point = vector_space @ plane_point.T
        plane_points.append(plane_point)
    return np.array(plane_points)


def match_rows(array, match_array, tolerance=0):
    '''
    Given an array, check if each row is in a secondary match array. Returns a
    boolean array with True for rows that appear within the match array, False
    otherwise. Uses Numpy Broadcasting to achieve this.
    '''
    boolean_rows = np.isclose(match_array, array[:, None], atol=tolerance)
    boolean_rows = boolean_rows.all(axis=2).any(axis=1)
    return boolean_rows


def angle_between(vectors1, vectors2):
    '''
    Calculates every angle between one array of vectors and another; vectors1
    and vectors2 should be numpy arrays.

    If vectors1 = [A, B, C] and vectors2 = [D, E, F], where A-F are 3D vectors,
    the angle output will be the following:

    angles = [[Angle(A, D), Angle(A, E), Angle(A, F)],
              [Angle(B, D), Angle(B, E), Angle(B, F)],
              [Angle(C, D), Angle(C, E), Angle(C, F)]]
    '''
    vectors1 = np.array(vectors1)
    vectors2 = np.array(vectors2)
    angles = (vectors1 @ vectors2.T).T
    norm_axis = len(vectors1.shape)-1
    angles = angles/np.linalg.norm(vectors1, axis=norm_axis)
    angles = angles.T
    norm_axis = len(vectors2.shape)-1
    angles = angles/np.linalg.norm(vectors2, axis=norm_axis)
    angles = np.clip(angles, -1, 1)
    angles = np.arccos(angles)
    return angles


def group_by_angle(vectors, angle=0.01):
    '''
    Group an array of vectors by their angular separation. Vectors within the
    set angle of one another are grouped into a list of arrays. The list is
    returned, along with a list of arrays containing the corresponding indexes.
    '''
    groups = []
    index_groups = []
    indexes = np.array(range(vectors.shape[0]))
    vector_array = vectors.copy()
    while not vector_array.shape[0] == 0:
        vector = vector_array[0]
        angles = angle_between(vector_array, vector)
        group = vector_array[angles <= angle]
        groups.append(group)
        index_group = indexes[angles <= angle]
        index_groups.append(index_group)
        indexes = indexes[angles > angle]
        vector_array = vector_array[angles > angle]
    return (groups, index_groups)
