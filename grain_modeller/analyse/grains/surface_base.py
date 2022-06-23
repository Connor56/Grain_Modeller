'''
Name:
    Surface Base
Description:
    Contains base code for surface analysis algorithms.
'''
import numpy as np
from dataclasses import dataclass
import pandas as pd
import linear_algebra as linalg


@dataclass
class PlaneGroup():
    '''
    Holds a number of plane normals in the form of a numpy array.
    '''
    planes: np.ndarray
    name: str
    surface_energy: float = None
    area: float = 0
    captured_planes: list = None


@dataclass
class Simplex():
    '''
    Represents a simple triangle in a 3D space, making up one of many simple
    polygon units that form a convex hull.
    '''
    points: np.ndarray
    vectors: np.ndarray = None
    area: float = None
    normal: np.ndarray = None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (np.all(self.points == other.points) and
                np.all(self.vectors == other.vectors) and
                self.area == other.area and
                np.all(self.normal == other.normal))


@dataclass
class Surface():
    '''
    Stores all the simplexes of a surface, the total area of the surface, the
    volume it encloses, and can be used to group simplexes with similar
    face normals. By providing a list of expected planes, the surface can have
    its grouped simplexes regrouped into exact plane surfaces.
    '''
    simplexes: list
    area: float
    volume: float
    grouped_simplexes: list = None
    grouped_area: pd.DataFrame = None
    grouped_plane_area: pd.DataFrame = None


def get_surface(shell):
    '''
    Uses a pyvista produced PolyData object representing the shell of a grain
    to produce a surface object holding all it's simplexes and face normals.
    '''
    faces = get_faces(shell.faces)
    points = shell.points
    simplexes = []
    for face in faces:
        face_points = points[face]
        simplexes.append(get_simplex(face_points))
    return Surface(
        simplexes, np.around(shell.area, 8), np.around(shell.volume, 8))


def get_faces(face_array):
    '''
    Turns the standard face_array into a list of lists, where each row has the
    indices of the points making the face.
    '''
    faces = []
    index = 0
    while index < face_array.shape[0]-1:
        number_of_points = face_array[index]
        if number_of_points > 3:
            raise ValueError(
                f"Number of points: {number_of_points} greater than 3")
        index += 1
        point_indexes = face_array[index:index+number_of_points]
        faces.append(point_indexes)
        index += number_of_points
    return faces


def get_simplex(points):
    '''
    Creates a simplex object from a set of points.
    '''
    points = np.array(points)
    vectors = points[1:, :] - points[0]
    area = np.around(np.linalg.norm(np.cross(vectors[0], vectors[1]))/2, 8)
    normal = np.cross(vectors[0], vectors[1])
    unit_normal = normal/np.linalg.norm(normal)
    return Simplex(points, vectors, area, unit_normal)


def group_simplexes(simplexes, angle):
    '''
    Groups a list of simplexes by their normal, returns the groups as a list.
    '''
    normals = np.array([x.normal for x in simplexes])
    grouped_normals, index_groups = linalg.group_by_angle(normals, angle)
    simplexes = np.array(simplexes)
    grouped_simplexes = []
    for group in index_groups:
        grouped_simplexes.append(simplexes[group])
    return grouped_simplexes


def get_areas(grouped_simplexes):
    '''
    Takes in a list of grouped simplexes, using the list creates a pandas data
    frame storing the area of each simplex group.
    '''
    areas = []
    for group in grouped_simplexes:
        normal = group[0].normal
        none_zero_normal_values = normal[np.abs(normal) > 0]
        absolute_normal_values = np.abs(none_zero_normal_values)
        max_normal_value = np.max(absolute_normal_values)
        normal = normal/max_normal_value
        area = np.sum([x.area for x in group])
        areas.append({'Normal': normal, 'Area': area})
    areas = pd.DataFrame(areas)
    return areas


def group_by_plane(grouped_normals, plane_groups, angle):
    '''
    Finds the surface area of a given group of planes by comparing the vectors
    of grouped normals with the plane normals given in the plane groups.

    Uses a set of grouped_normals in the form of a pandas dataframe, produced
    by get_areas() and stored in Surface.grouped_area to perform this grouping.

    grouped_normals: Pandas dataframe containing normals in cartesian
        coordinates and their associated areas. Should be in the form given by
        get_areas().
    plane_groups: List of PlaneGroups to match normals with. PlaneGroups
        contain the name of a type of plane and the cartesian normals
        associated with it.
    angle: The allowed angle between the cartesian planes defined in the plane
        group, and those in the grouped_normals dataframe.
    '''
    editable_grouped_normals = grouped_normals.copy()
    for plane_group in plane_groups:
        normals = editable_grouped_normals['Normal'].values.tolist()
        normals = np.array(normals).astype(float)
        angles = linalg.angle_between(normals, plane_group.planes)
        within_range = np.any(angles < angle, axis=1)
        normals_in_range = editable_grouped_normals[within_range]
        plane_group.area = np.sum(normals_in_range['Area'])
        plane_group.captured_planes = (
            normals_in_range['Normal'].values.tolist())
        editable_grouped_normals = editable_grouped_normals[
            np.invert(within_range)]
        if editable_grouped_normals.shape[0] < 1:
            break
    total_area = np.sum(grouped_normals['Area'])
    plane_area = sum([x.area for x in plane_groups])
    remaining_area = total_area-plane_area
    remaining_planes = editable_grouped_normals['Normal'].values.tolist()
    if len(remaining_planes) < 1:
        remaining_planes = 'None'
    plane_data = pd.DataFrame(
        {'Name': [x.name for x in plane_groups]+['Remainder'],
         'Plane_Group': [x.planes for x in plane_groups]+['None'],
         'Area': [x.area for x in plane_groups]+[remaining_area],
         'Captured_Planes':
            [x.captured_planes for x in plane_groups]+[remaining_planes]
         })
    return plane_data
