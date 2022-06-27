'''
Name:
    Edits
Description:
    Contains functions for editing supercells, for example removing atoms, or
    other kinds of none reversible changes.
'''

import numpy as np
import linear_algebra as linalg
import copy
from dataclasses import dataclass
import transforms
import crystallography
import testing_tools as test_tool


class Cut():

    def __init__(self, cut_type, point, plane=None, radius=None, out=True,
                 axes=None):
        '''
        Initialises the cut object, all cuts have a type: 's' for spherical,
        'p' for plane, 'e' for ellipsoid, and a point in space. The optional
        arguments are variables required by the different types. Give plane in
        miller indices, axes are the axis, semi major, and semi minor axis of
        an ellipsoid. Out defines if points that are left from an 's' or 'e'
        cut are outside or inside of the sphere/ ellipsoid.
        '''
        if not cut_type in ['s', 'p', 'cp', 'e']:
            raise ValueError(f"Unknown cut type: '{cut_type}'.")
        if not isinstance(point, list) and not isinstance(point, np.ndarray):
            raise ValueError(f"Point: '{point}', is not a list or numpy "
                             "array.")
        if len(point) != 3:
            raise ValueError(f'Point: {point}, is not 3D.')
        self.cut_type = cut_type
        self.point = np.array(point)
        self.plane = np.array(plane) if cut_type in ['p', 'cp'] else plane
        self.radius = radius
        self.out = out
        self.axes = np.array(axes) if cut_type == 'e' else axes

    def __repr__(self):
        plane = self.plane
        if self.cut_type == 'p':
            plane = self.plane.tolist()
        return (f"Cut('{self.cut_type}', {self.point.tolist()}, "
                f"plane={plane}, radius={self.radius}, out={self.out}, "
                f"axes={self.axes})")


@dataclass
class Reflection():
    '''
    Stores a plane as a point and a normal, this can then be used to reflect
    atoms of a supercell across the plane. The out variable decides whether
    atoms on the positive side of the plane, or negative side of the plane
    will be deleted in the reflection. Define the normal and point in
    fractional coordinates, they will be transformed into their cartesian
    counterparts during the execution of the reflection.
    '''
    point: np.ndarray
    normal: np.ndarray
    out: bool = True


@dataclass
class Twin():
    '''
    Stores all the information required to create a twin by rotation across
    a plane along some rotation vector. Normalises the normal vector and
    rotation vector passed to it by default. To switch off this behaviour set
    normalise_vectors to False. Angle should be in radians, and the point,
    normal, and rotation_vector should be in 3D.
    '''
    point: np.ndarray
    normal: np.ndarray
    rotation_vector: np.ndarray
    angle: float
    normalise_vectors: bool = True

    def __post_init__(self):
        angle = linalg.angle(self.normal, self.rotation_vector)
        if not np.isclose(angle, np.pi/2, atol=1e-8):
            raise ValueError(
                f"Normal vector and rotation vector must be orthogonal. "
                f"Vectors {self.normal} and {self.rotation_vector} are not "
                "orthogonal.")

        if not np.isclose(np.linalg.norm(self.normal), 1, atol=1e-6):
            if self.normalise_vectors:
                Warning("normal has been made a unit vector.")
                self.normal = linalg.normalise(self.normal)
            else:
                Warning(f"Normal {self.normal} is not a unit vector. This may "
                        "cause issues whilst creating the twin.")

        if not np.isclose(np.linalg.norm(self.rotation_vector), 1, atol=1e-6):
            if self.normalise_vectors:
                Warning("rotation_vector has been made a unit vector.")
                self.rotation_vector = linalg.normalise(self.rotation_vector)
            else:
                Warning(f"Rotation Vector {self.rotation_vector} is not a "
                        "unit vector. This may cause issues whilst creating "
                        "the twin.")


def make_cut(supercell, cut):
    '''
    Directs different cut types to the correct function.
    '''
    if cut.cut_type == 'p': plane_cut(supercell, cut)
    elif cut.cut_type == 'cp': cartesian_plane_cut(supercell, cut)
    elif cut.cut_type == 's': spherical_cut(supercell, cut)
    elif cut.cut_type == 'e': ellipsoid_cut(supercell, cut)
    if supercell.cartesian is not None: supercell.set_cartesian()


def plane_cut(supercell, cut):
    '''
    Cuts a supercell along a plane, given by a plane miller index and a point
    in the plane defined in the cut variable. Atoms on the positive side of the
    miller index are deleted. Uses fractional coordinates, so plane points
    should be given in fractional also.
    '''
    miller_indexes = np.array(cut.plane).astype(float)
    fractional_vector_space = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    plane_normal = crystallography.cartesian_plane_normal(
        fractional_vector_space, miller_indexes)
    point = np.array(cut.point)
    atoms = supercell.fractional
    atoms = atoms['coordinates']
    atoms = atoms-point
    deletions = np.dot(atoms, plane_normal)
    deletions = (deletions < 0.0001)
    supercell.fractional = supercell.fractional[deletions]


def cartesian_plane_cut(supercell, cut):
    '''
    Cuts a supercell along a plane, given by a plane miller index and a point
    in the plane defined in the cut variable. Atoms on the positive side of the
    miller index are deleted. Uses cartesian coordinates, so plane points and
    "miller indices" correspond to the x, y, z directions, not to the a, b, c
    directions of the supercell itself. Thus you must give points in cartesian
    coordinates, and normal directions in cartesian coordinates also.
    '''
    #miller_indexes = np.array(cut.plane)
    plane_normal = np.array(cut.plane)
    #print(miller_indexes)
    #print(supercell.vector_space)
    #plane_normal = crystallography.cartesian_plane_normal(
    #    supercell.vector_space, miller_indexes)
    #print(plane_normal)
    point = np.array(cut.point)
    if supercell.cartesian is None: supercell.set_cartesian()
    atoms = supercell.cartesian
    atoms = atoms['coordinates']
    atoms = atoms-point
    #print(plane_normal)
    deletions = np.dot(atoms, plane_normal)
    deletions = (deletions < 0.0001)
    supercell.fractional = supercell.fractional[deletions]


def spherical_cut(supercell, cut):
    '''
    Makes a spherical cut in a grain, using a given point and radius, cut has
    an out value which dictates whether points outside the sphere remain or
    inside the sphere remain. Points remaining outside the sphere is default.
    '''
    atoms = supercell.fractional
    atoms = atoms['coordinates']
    atoms = np.linalg.norm(atoms-cut.point, axis=1)
    deletions = atoms > cut.radius if cut.out else atoms < cut.radius
    supercell.fractional = supercell.fractional[deletions]


def reflect(supercell, reflection):
    '''
    Reflects a supercell across the plane given in a reflection dataclass.
    Removes atoms on one side of the reflection plane to prevent the reflection
    from occuring across both sides and potentially ruining the supercell
    structure. Uses cartesian coordinates to perform the reflection, as
    fractional coordinates rely on vector space which cannot be reflected
    across the plane.
    '''
    cut = Cut('p', reflection.point, reflection.normal)
    make_cut(supercell, cut)
    supercell.set_cartesian()
    atoms = supercell.cartesian
    cartesian_point = supercell.vector_space @ reflection.point
    plane_distances = cartesian_point-atoms['coordinates']
    normal = crystallography.cartesian_plane_normal(
        supercell.vector_space, reflection.normal)
    reflection_distances = np.dot(plane_distances, normal)*2
    reflection_vectors = np.outer(reflection_distances, normal)
    reflection_vectors = np.around(reflection_vectors, 8)
    reflection_atoms = np.copy(atoms)
    reflection_atoms['coordinates'] += reflection_vectors
    in_plane_atoms = np.isclose(reflection_distances, 0,  atol=1e-05)
    in_plane_atoms = np.invert(in_plane_atoms)
    reflection_atoms = reflection_atoms[in_plane_atoms]
    atoms = np.hstack((atoms, reflection_atoms)).flatten()
    supercell.cartesian = atoms
    supercell.set_fractional()


def create_twin(supercell, twin):
    '''
    Uses a twin dataclass to rotate a supercell around a rotation vector that
    lies within a given plain. In order for this to work well the rotation
    vector should be chosen carefully, generally so that it points along some
    line of symmetry that runs parallel to the plane.
    '''
    rotated_supercell = copy.deepcopy(supercell)
    transforms.translate(rotated_supercell, -twin.point)
    transforms.rotate(supercell, twin.rotation_vector)
