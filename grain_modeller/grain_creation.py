'''
Module:
    Grain Creation
Description:
    This module contains standalone functions for creating the input for
    atomistic simulations. The idea is to create grains using series of cuts to
    a given structure, then modify the surface of that structure to make all
    the different grains stoichiometrically identical.
'''

from scipy import spatial
import numpy as np
import pandas as pd
from atom import Atom
from unitcell import UnitCell
from supercell import SuperCell
import edits
import testing_tools as test_tool
import copy


class Grain():

    def __init__(self, name, unitcell, cuts, repeat_ratio):
        '''
        A single cut defines the miller index of a cut plane, and a point in
        that plane. Because grains can be of arbitrary size, you should give
        the plane point in fractional coordinates. Cuts define final shape.
        Repeat ratios are the number of repeats a unitcell should have in the
        x, y, and z direction. Cuts should be of form given in edits.py e.g.:

                Cut('p', [1, 1, 1], plane=[1, 1, 1]) or
                Cut('s', [0, 1, 0], radius=5)
                Cut('e', [1, 1, 0], axes=[0.4, 0.2, 1])

        Repeat ratios alter cuts thusly: if you have a cut like the first one
        given above: Cut('p', [0, 1, 1], plane=[1, 1, 1]) with repeats
        [1, 1, 2], and you make this grain 10x bigger, you'll have a 10x10x20
        cuboid supercell, which you cut once with the [1, 1, 1] plane at the
        [0, 10, 20] point. So the point is repeats*10*fractional_positions.
        This works fine for ellipsoid and plane cuts, however be careful when
        using spherical cuts, they will not follow the repeat ratio shape.
        Instead their radius will scale with the smallest repeat ratio,
        although their point will scale with the repeat ratios.
        '''
        self.name = name
        self.unitcell = unitcell
        self.cuts = cuts
        self.repeat_ratio = np.array(repeat_ratio)
        self.border = 0
        self.scale_factor = None
        self.supercell = None
        self.best_composition = None
        self.distance_symmetries = None
        self.surface_atoms = None
        self.composition_deltas = None
        self.matched_supercell = None

    def __repr__(self):
        repeat_ratio = self.repeat_ratio.tolist()
        return (f"Grain('{self.name}', {self.unitcell}, {self.cuts}, "
                + f"{repeat_ratio})")


def compositionally_match(grains, atom_target):
    '''
    Produces compositionally matched grains from grain shape definitions. All
    grains will be matched so that they have exactly the same elemental
    composition, provided they are "complementary". Two grains can be
    considered complementary if their underlying paramters (unitcell, shape)
    allow them in principle to be compositionally idenitcal - effectively both
    can be reached from the other by rearrangement of the same group of atoms.

    An example of two grains which are complementary: two separate shapes made
    from the same or suitably similar unitcells. An example of two
    uncomplementary grains: two grains with unitcells containing completely
    different elements, or the same elements at very different ratios; in
    either case it is not possible in principle to rearrange one into the
    other.
    '''
    # Single grain input
    if type(grains) != list:
        grains = [grains]
    grains = scale_grains(grains, atom_target)
    grains = best_composition(grains, atom_target)
    for grain in grains:
        grain = remove_surface_atoms(grain)
    return grains


def scale_grains(grains, atom_target):
    '''
    Produces a cuboid supercell from an underlying unitcell, using the provided
    grain list, each entry of which is a definition object for a grain shape.
    Makes cuts into this cuboid using a list of cuts defined in the grain,
    producing a final grain shape. Each grain in the grain list is scaled until
    its total number of atoms is as close as possible to the given atom target
    whilst also being greater than that atom target. If scaling fails then the
    scale factor of the grain will be iteratively increased by one until the
    number of atoms is above the atom_target.

    Grain shape definitions include the unitcells that produce them.
    '''
    for grain in grains:
        columns = [('scale_factor', 'f8'), ('number_of_atoms', 'f8'),
                   ('supercell', object)]
        previous_grains = np.array([], dtype=columns)
        scale_factor = 2
        scale = True
        while True:
            supercell = build_grain(grain, scale_factor)
            number_of_atoms = supercell.fractional.shape[0]
            grain_record = np.array((scale_factor, number_of_atoms, supercell),
                                    dtype=columns)
            previous_grains = np.append(previous_grains, grain_record)
            factor_multiple = (1/(number_of_atoms/atom_target))**(1/3)
            scale_factor = int(scale_factor*factor_multiple)
            number_of_atoms = previous_grains['number_of_atoms']
            min_reached = np.max(number_of_atoms) > atom_target
            scale_factors = previous_grains['scale_factor']
            if scale_factor in scale_factors and min_reached:
                break
            elif scale_factor in scale_factors and not min_reached:
                scale_factor += 1
                scale = False
            elif not scale and not min_reached:
                scale_factor += 1
            elif not scale and min_reached:
                break
        viable_grains = previous_grains['number_of_atoms'] > atom_target
        previous_grains = previous_grains[viable_grains]
        sort_order = np.argsort(previous_grains['number_of_atoms'])
        grain.scale_factor = scale_factor
        grain.supercell = previous_grains[sort_order[0]]['supercell']
    return grains


def build_grain(grain, scale_factor):
    '''
    Builds a grain based on a grain object and a size factor which determines
    scale.
    '''
    x_repeat = grain.repeat_ratio[0]*scale_factor
    y_repeat = grain.repeat_ratio[1]*scale_factor
    z_repeat = grain.repeat_ratio[2]*scale_factor
    supercell = SuperCell(grain.unitcell, x_repeat, y_repeat, z_repeat)
    cuts = alter_cuts(scale_factor, grain)
    supercell = cut_grain(supercell, cuts)
    grain.supercell = supercell
    return supercell


def alter_cuts(scale_factor, grain):
    '''
    Alter the plane point positions in grain definition's cut list, so that
    the shape scales with the size factor used to resize the grain.
    '''
    repeats = scale_factor*grain.repeat_ratio
    cuts = copy.deepcopy(grain.cuts)
    for cut in cuts:
        cut.point = cut.point*repeats
        if cut.cut_type == 's': cut.radius *= np.min(repeats)
    return cuts


def cut_grain(supercell, cuts):
    '''
    Uses a given list of cuts to delete atoms from a given supercell. Cuts must
    be in the form of a sequence, with each entry being a Cut object defined in
    the edits.py module.
    '''
    for cut in cuts:
        edits.make_cut(supercell, cut)
    return supercell


def best_composition(grains, atom_target):
    '''
    Defines the composition all grains should be fit to, taking into account
    the elemental ratios of the different grains. A best fit composition is one
    which can be reached by all grains.
    '''
    columns = composition_columns(grains[0], float)
    columns_integer = composition_columns(grains[0], int)
    ratios = np.array([], dtype=columns)
    for grain in grains:
        elements = grain.supercell.fractional['element']
        unique, counts = np.unique(elements, return_counts=True)
        counts = counts/elements.shape[0]
        ratios = np.append(ratios, np.array(tuple(counts), dtype=columns))
    composition = [int(np.mean(ratios[element])*atom_target) for element
                   in ratios.dtype.names]
    additions = [0]*len(composition)
    for x in range(atom_target-sum(composition)):
        index = x % len(additions)
        additions[index] += 1
    composition = np.array(composition) + np.array(additions)
    composition = np.array(tuple(composition), dtype=columns_integer)
    composition = pd.DataFrame(composition, index=[0])
    for grain in grains:
        grain.best_composition = composition
    return grains


def composition_columns(grain, c_type):
    '''
    Produces composition columns for the structured arrays which hold element
    compositions, type defines numerical type. For some cases only the dtype
    of the columns is returned, for others the composition values are returned
    also.
    '''
    elements = grain.supercell.fractional['element']
    elements, counts = np.unique(elements, return_counts=True)
    if c_type == float:
        columns = [(element, 'f8') for element in elements]
    elif c_type == int:
        columns = [(element, 'i8') for element in elements]
    else:
        raise ValueError('Type is not recognised.')
    return columns


def remove_surface_atoms(grain):
    '''
    Deletes atoms from the surface of a supercell until the supercell matches a
    given composition. Proceeding in rounds, the most extreme surface atoms of
    each round are deleted. Extreme meaning those that would physically have
    the lowest bonding strength. Surface atoms are found first, because this
    significantly increases the speed of ConvexHull at larger sizes.
    '''
    print(grain.name)
    while True:
        print('Getting New Surface')
        grain = get_surface_atoms(grain)
        surface_atoms = grain.surface_atoms
        current_composition = get_composition(grain)
        composition_deltas = current_composition - grain.best_composition
        grain.composition_deltas = composition_deltas
        while True:
            try:
                vertices = spatial.ConvexHull(surface_atoms['coordinates'])
            except spatial.qhull.QhullError:
                break
            vertices = vertices.vertices
            if remove_atoms(grain, surface_atoms, vertices):
                print(grain.composition_deltas)
                break
            removed = np.invert(grain.surface_atoms['removed'])
            surface_atoms = grain.surface_atoms[removed]
            print(grain.composition_deltas)
            if surface_atoms.shape[0] < 6:
                break
        # Delete atoms at random if deletions make up less than X% of surface
        if update_supercell(grain):
            surface_percent = np.sum(grain.composition_deltas.values)
            surface_percent /= grain.surface_atoms.shape[0]
            if surface_percent < 0.04:
                vertices = np.arange(grain.surface_atoms.shape[0])
                vertices = vertices[np.invert(grain.surface_atoms['removed'])]
                remove_atoms(grain, surface_atoms, vertices)
                update_supercell(grain)
            else:
                raise ValueError(
                    "It's not possible to compositionally match these grains."
                    + f" The composition of grain: {grain.name}, cannot match "
                    + "the best composition. Try a different size or a "
                    + "different combination of grains.")
        if np.sum(grain.composition_deltas.values) == 0:
            break
    return grain


def get_surface_atoms(grain):
    '''
    Finds the surface atoms of a given grain. Uses the fact crystals are
    regular to establish general patterns for different atom types in the
    crystal bulk. Any atoms that fall out of this pattern are deemed surface
    atoms. The pattern for the bulk crystal is found by creating a cubic
    supercell and finding the different possible patterns the atoms can have in
    the bulk.
    '''
    if grain.distance_symmetries is None:
        supercell = SuperCell(grain.supercell.unitcell, 10, 10, 10)
        atoms = supercell.fractional['coordinates']
        atom_tree = spatial.cKDTree(atoms)
        distances, indexes = atom_tree.query(atoms, k=13)
        bulk = np.all((atoms <= 9) & (atoms >= 1), axis=1)
        distances = np.around(distances[bulk], 6)
        distances = np.unique(distances, axis=0)
        grain.distance_symmetries = distances[:, 1:]
    atoms = grain.supercell.fractional['coordinates']
    atom_tree = spatial.cKDTree(atoms)
    distances, indexes = atom_tree.query(atoms, k=13)
    distances = np.around(distances, 6)
    distances = distances[:, 1:]
    distances = distances[:, None]
    # Creates a mask for the surface atoms in the array.
    surface = (grain.distance_symmetries == distances).all(axis=2).any(1)
    surface_atoms = grain.supercell.fractional[np.invert(surface)]
    dtypes = [('element', 'U10'), ('coordinates', 'f8', 3), ('index', 'i8'),
              ('removed', '?')]
    grain.surface_atoms = np.zeros(surface_atoms.shape[0], dtype=dtypes)
    grain.surface_atoms['element'] = surface_atoms['element']
    grain.surface_atoms['coordinates'] = surface_atoms['coordinates']
    grain.surface_atoms['index'] = np.arange(len(surface))[np.invert(surface)]
    grain.surface_atoms['removed'] = [False]*surface_atoms.shape[0]
    return grain


def get_composition(grain):
    '''
    Get the elemental composition of the given grain.
    '''
    elements = grain.supercell.fractional['element']
    elements, counts = np.unique(elements, return_counts=True)
    composition = [(element, 'i8') for element in elements]
    composition = np.array(tuple(counts), dtype=composition)
    composition = pd.DataFrame(composition, index=[0])
    return composition


def remove_atoms(grain, surface_atoms, vertices):
    '''
    Register which surface atoms have been removed. The grain shape object
    contains an internal surface_atoms variable which has a column that records
    removal; using indexes, atoms will be recorded as removed there.
    '''
    vertex_indices = surface_atoms[vertices]['index']
    vertex_indices = (grain.surface_atoms['index'][:, None] == vertex_indices)
    vertex_indices = vertex_indices.any(1)
    elligible_atoms = grain.surface_atoms[vertex_indices]
    composition_before = copy.deepcopy(grain.composition_deltas)
    for element in grain.composition_deltas:
        atoms = elligible_atoms[elligible_atoms['element'] == element]
        to_delete = grain.composition_deltas[element][0]
        indexes = atoms['index']
        indexes = (grain.surface_atoms['index'][:, None] == indexes).any(1)
        # Case statement has to be used because of masking complexities
        if len(atoms) == 0:
            delete = 0
            continue
        elif len(atoms) >= to_delete:
            remove = [True]*to_delete + [False]*(len(atoms)-to_delete)
            delete = to_delete
            grain.surface_atoms['removed'][indexes] = remove
        elif len(atoms) < to_delete:
            delete = len(atoms)
            grain.surface_atoms['removed'][indexes] = True
        grain.composition_deltas[element][0] -= delete
    if np.all(composition_before == grain.composition_deltas):
        return True
    elif np.sum(grain.composition_deltas.values) == 0:
        return True
    else:
        return False


def update_supercell(grain):
    '''
    Update the grain's supercell, by deleting all the atoms marked for removal
    in its surface atom array.
    '''
    removed_indexes = grain.surface_atoms[grain.surface_atoms['removed']]
    if removed_indexes.tolist() == []:
        return True
    removed_indexes = removed_indexes['index']
    atom_range = np.arange(grain.supercell.fractional.shape[0])
    removed = np.invert((atom_range[:, None] == removed_indexes).any(1))
    grain.supercell.fractional = grain.supercell.fractional[removed]
    return False
