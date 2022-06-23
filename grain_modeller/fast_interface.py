import fast_grain as fg
import numpy as np
import grain_analysis_hull as gah
import fast_transforms as ft
import testing_tools as test_tool
from collections import namedtuple
from scipy import spatial
import grain_creation as gc
import interface


def setup_interface(interface_outline):
    '''
    Takes in an outline of the interface to be created, and uses it to
    produce the desired interface.

    interface_outline: A namedtuple of information for the interface, including
        unitcell_1, plane_1, unitcell_2, plane_2, and desired_depth.
    return: A lammps data file of the desired simulation type.
    '''
    #Create First Interface
    #Make a Slab
    supercell_1 = fg.FastSuperCell(
        interface_outline.unitcell_1, interface_outline.repeat_1,
        interface_outline.repeat_1, interface_outline.repeat_1)
    unitcell_1_attrib = interface_outline.unitcell_1.get_attributes()
    unitcell_1_centre = (
        np.array(unitcell_1_attrib['a_Lattice_Vector'])
        + np.array(unitcell_1_attrib['b_Lattice_Vector'])
        + np.array(unitcell_1_attrib['c_Lattice_Vector']))*(
        interface_outline.repeat_1/2)
    #Make First Cut
    supercell_1.make_cut_fast(interface_outline.plane_1, unitcell_1_centre)
    test_tool.xyz_file_output('supercell_1_111_first_cut',
        supercell_1.get_atom_array())
    #Get Atom For Desired Depth of Secondd cut
    normal_atoms = ft.get_normal_atoms(
        supercell_1, -interface_outline.plane_1, unitcell_1_centre)
    normal_mask = (np.linalg.norm(normal_atoms - unitcell_1_centre, axis=1)
                   > interface_outline.desired_depth)
    second_cut_atom = normal_atoms[normal_mask][0]
    #Make Second Cut
    supercell_1.make_cut_fast(-interface_outline.plane_1, second_cut_atom)
    #Make Third Cut to remove surface atoms
    normal_1 = supercell_1.create_normal(interface_outline.plane_1)
    supercell_1.make_cut_fast(
        interface_outline.plane_1,
        unitcell_1_centre-(normal_1*interface_outline.third_cut_depth))
    #Rotate the Slab
    supercell_1 = rotate_atoms(supercell_1, normal_1)
    supercell_1 = cut_out_cube(supercell_1)
    test_tool.xyz_file_output('supercell_1_cut', supercell_1.get_atom_array())

    #Create Second Interface
    #Make a Slab
    supercell_2 = fg.FastSuperCell(
        interface_outline.unitcell_2, interface_outline.repeat_2,
        interface_outline.repeat_2, interface_outline.repeat_2)
    unitcell_2_attrib = interface_outline.unitcell_2.get_attributes()
    unitcell_2_centre = (
        np.array(unitcell_2_attrib['a_Lattice_Vector'])
        + np.array(unitcell_2_attrib['b_Lattice_Vector'])
        + np.array(unitcell_2_attrib['c_Lattice_Vector']))*(
        interface_outline.repeat_2/2)
    #Make First Cut
    supercell_2.make_cut_fast(-interface_outline.plane_2, unitcell_2_centre)
    #Get Atom For Desired Depth of Second cut
    normal_atoms = ft.get_normal_atoms(
        supercell_2, interface_outline.plane_2, unitcell_2_centre)
    normal_mask = (np.linalg.norm(normal_atoms - unitcell_2_centre, axis=1)
                   > interface_outline.desired_depth)
    second_cut_atom = normal_atoms[normal_mask][0]
    #Make Second Cut
    supercell_2.make_cut_fast(interface_outline.plane_2, second_cut_atom)
    #Rotate the Slab
    normal_2 = supercell_2.create_normal(-interface_outline.plane_1)
    supercell_2 = rotate_atoms(supercell_2, normal_1)
    supercell_2 = cut_out_cube(supercell_2)
    test_tool.xyz_file_output('supercell_2_cut', supercell_2.get_atom_array())

    supercell_1, supercell_2 = match_cubes(
        supercell_1, supercell_2, interface_outline.spacing_x,
        interface_outline.length_x, interface_outline.spacing_y,
        interface_outline.length_y, interface_outline.num_z_distances)

    create_interface_simulation(supercell_1, supercell_2, interface_outline)






def rotate_atoms(supercell, normal):
    '''
    Rotate atoms so that normal points in the z direction.

    supercell: FastSuperCell object
    normal: Numpy array of the plane normal.
    return: The rotated supercell.
    '''
    atom_array = supercell.get_atom_array()
    atoms = np.vstack((atom_array['x'], atom_array['y'], atom_array['z'])).T
    if normal[1] == 0:
        z_rotation = np.pi/2
    else:
        z_rotation = np.arctan(normal[0]/normal[1])
    normal = gah.rotate_plane_atoms(normal, z_rotation, 'z')
    atoms = gah.rotate_plane_atoms(atoms, z_rotation, 'z')
    if normal[2] == 0:
        x_rotation = np.pi/2
    else:
        x_rotation = np.arctan(normal[1]/normal[2])
    normal = gah.rotate_plane_atoms(normal, x_rotation, 'x')
    atoms = gah.rotate_plane_atoms(atoms, x_rotation, 'x')
    atom_array['x'] = atoms[:, 0]
    atom_array['y'] = atoms[:, 1]
    atom_array['z'] = atoms[:, 2]
    supercell.set_atom_array(atom_array)
    return supercell


def cut_out_cube(supercell):
    '''
    Cuts out a cubes from a given supercell atom array.

    supercell: FastSuperCell object
    return: FastSuperCell with atoms deleted to make it a cube.
    '''
    atom_array = supercell.get_atom_array()
    atoms = np.vstack((atom_array['x'], atom_array['y'], atom_array['z'])).T
    atom_mins = np.min(atoms, axis=0)
    atoms = atoms - atom_mins
    atom_array['x'] = atoms[:, 0]
    atom_array['y'] = atoms[:, 1]
    atom_array['z'] = atoms[:, 2]
    supercell.set_atom_array(atom_array)
    Cut = namedtuple('Cut', ['Point', 'Normal'])
    cut_normals = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                   np.array([0, 1, 0]), np.array([0, -1, 0])]
    desired_density = atoms.shape[0]/spatial.ConvexHull(atoms).volume
    print(desired_density)
    density = 0
    cut_amount = 5
    while density < 0.999*desired_density:
        atom_array = supercell.get_atom_array()
        atoms = np.vstack(
            (atom_array['x'], atom_array['y'], atom_array['z'])).T
        min_coords = np.min(atoms, axis=0)
        max_coords = np.max(atoms, axis=0)
        volume = np.prod(max_coords-min_coords)
        density = atom_array.shape[0]/volume
        print(density, atom_array.shape)
        cut_list = [Cut([min_coords[0]+cut_amount, 0, 0], [-1, 0, 0]),
                    Cut([max_coords[0]-cut_amount, 0, 0], [1, 0, 0]),
                    Cut([0, min_coords[1]+cut_amount, 0], [0, -1, 0]),
                    Cut([0, max_coords[1]-cut_amount, 0], [0, 1, 0])]
        gc.cut_grain(supercell, cut_list)
    return supercell


def match_cubes(supercell_1, supercell_2, spacing_x, length_x, spacing_y,
                length_y, num_z_distances):
    '''
    Takes in two supercell cubes and using the given spacing in x and y,
    searches along the given lengths in x and y, along with z to find the
    minimum energy position for the two slabs relative to one another.

    supercell_1: FastSuperCell object of first cube supercell.
    supercell_2: FastSuperCell object of second cube supercell.
    spacing_x: The size of the search step to move along in x.
    length_x: The size to be split up into steps along x.
    spacing_y: The size of the search step to move along in y.
    num_z_distances: The number of distances in the z direction to try, the
        distances are gained from finding the spacing between atoms in the
        larger supercell in the z direction and matching it to that. The more
        distances chosen the longer the fitting.
    return: The two supercells now matched to one another.
    '''
    #Shift both cubes to 0, 0, 0
    #Supercell 1 shift
    atoms_1 = supercell_1.get_atom_array_cartesian()
    atoms_1_mins = np.min(atoms_1, axis=0)
    atoms_1_maxes = np.max(atoms_1, axis=0)
    area_1 = np.prod((atoms_1_maxes-atoms_1_mins)[:2])
    supercell_1.translate_atoms(-atoms_1_mins)
    #Supercell 2 shift
    atoms_2 = supercell_2.get_atom_array_cartesian()
    atoms_2_mins = np.min(atoms_2, axis=0)
    atoms_2_maxes = np.max(atoms_2, axis=0)
    area_2 = np.prod((atoms_2_maxes-atoms_2_mins)[:2])
    supercell_2.translate_atoms(-atoms_2_mins)
    test_tool.xyz_file_output('supercell_1_translated', supercell_1.get_atom_array())
    test_tool.xyz_file_output('supercell_2_translated', supercell_2.get_atom_array())
    print('Area 1: ', area_1, 'Area 2: ', area_2)
    #Get Average neighbour distances
    if area_1 > area_2:
        atoms_1 = supercell_1.get_atom_array_cartesian()
        atoms_1_tree = spatial.cKDTree(atoms_1)
        average_distances = np.around(atoms_1_tree.query(atoms_1, k=13)[0], 5)
        unique_distances = average_distances[:, 1:]
        unique_distances = np.unique(unique_distances)
        occurence_list = []
        for distance in unique_distances:
            occurence = np.sum(average_distances == distance)
            occurence_list.append((occurence, distance))
        average_distance = (np.sum([
            distance[0]*distance[1] for distance in occurence_list])
            / np.sum(np.array(occurence_list)[:, 0]))
    else:
        atoms_2 = supercell_2.get_atom_array_cartesian()
        atoms_2_tree = spatial.cKDTree(atoms_2)
        average_distances = np.around(atoms_2_tree.query(atoms_2, k=13)[0], 5)
        unique_distances = average_distances[:, 1:]
        unique_distances = np.unique(unique_distances)
        occurence_list = []
        for distance in unique_distances:
            occurence = np.sum(average_distances == distance)
            occurence_list.append((occurence, distance))
        average_distance = (np.sum([
            distance[0]*distance[1] for distance in occurence_list])
            / np.sum(np.array(occurence_list)[:, 0]))

    atoms_1 = supercell_1.get_atom_array_cartesian()
    z_shift = np.array([0, 0, np.max(atoms_1, axis=0)[2]])
    supercell_2.translate_atoms(z_shift)
    test_tool.xyz_file_output(
        'supercell_2_shifted_up', supercell_2.get_atom_array())
    #Test Supercell configurations
    if area_1 > area_2:
        atoms_1_tree = spatial.cKDTree(atoms_1)
        central_position = np.mean(atoms_1, axis=0)
        central_index = atoms_1_tree.query(central_position)[1]
        central_atom = atoms_1[central_index]
        z_distances = np.sort((atoms_1-central_atom)[:, 2])
        z_distances = np.around(z_distances[z_distances > 0.1], 4)
        z_distances = np.unique(z_distances)
        z_distances = z_distances[:num_z_distances]
        atoms_2 = supercell_2.get_atom_array_cartesian()
        atom_check_mask = atoms_2[:, 2] < (0.1+np.min(atoms_2, axis=0)[2])
        Shift = namedtuple('Shift', ['shift', 'average_off'])
        shift_preference = []
        #Shift list for moving cells relative to one another
        shift_list = np.array([
            [x*spacing_x, y*spacing_y, z]
            for x in range(0, int(length_x/spacing_x)+1)
            for y in range(0, int(length_y/spacing_y)+1)
            for z in z_distances])
        print(shift_list)
        #Find the best shifts by geometry at interface
        for shift in shift_list:
            print(shift)
            atoms_2_shifted = atoms_2+shift
            plane_atoms = atoms_2_shifted[atom_check_mask]
            interface_distances = atoms_1_tree.query(plane_atoms, k=13)[0]
            interface_distances = interface_distances[:, 1:]
            if np.any(interface_distances < average_distance*0.8):
                continue
            interface_distances = np.around(interface_distances, 5)
            unique_distances = np.unique(interface_distances)
            occurence_list = []
            for distance in unique_distances:
                occurence = np.sum(interface_distances == distance)
                occurence_list.append((occurence, distance))
            average_interface_distance = (np.sum([
                distance[0]*distance[1] for distance in occurence_list])
                / np.sum(np.array(occurence_list)[:, 0]))
            print('Average Off: ',
                  np.abs(average_interface_distance-average_distance))
            shift_preference.append(
                Shift(shift,
                      np.abs(average_interface_distance-average_distance)))
        shift_preference.sort(key=lambda x: x.average_off)
        best_shift = shift_preference[0]
        print('This is the best shift by averages: ', best_shift)
        supercell_2.translate_atoms(best_shift.shift)
    else:
        atoms_2 = supercell_2.get_atom_array_cartesian()
        atoms_2_tree = spatial.cKDTree(atoms_2)
        central_position = np.mean(atoms_2, axis=0)
        central_index = atoms_2_tree.query(central_position)[1]
        central_atom = atoms_2[central_index]
        z_distances = np.sort((atoms_2-central_atom)[:, 2])
        z_distances = np.around(z_distances[z_distances > 0.1], 4)
        z_distances = np.unique(z_distances)
        z_distances = z_distances[:num_z_distances]
        atoms_1 = supercell_1.get_atom_array_cartesian()
        atom_check_mask = atoms_1[:, 2] > (-0.1+np.max(atoms_1, axis=0)[2])
        Shift = namedtuple('Shift', ['shift', 'average_off'])
        shift_preference = []
        #Shift list for moving cells relative to one another
        shift_list = np.array([
            [x*spacing_x, y*spacing_y, -z]
            for x in range(0, int(length_x/spacing_x)+1)
            for y in range(0, int(length_y/spacing_y)+1)
            for z in z_distances])
        print(shift_list)
        #Find the best shifts by geometry at interface
        for shift in shift_list:
            print(shift)
            atoms_1_shifted = atoms_1+shift
            plane_atoms = atoms_1_shifted[atom_check_mask]
            interface_distances = atoms_2_tree.query(plane_atoms, k=13)[0]
            interface_distances = interface_distances[:, 1:]
            if np.any(interface_distances < average_distance*0.8):
                continue
            interface_distances = np.around(interface_distances, 5)
            unique_distances = np.unique(interface_distances)
            occurence_list = []
            for distance in unique_distances:
                occurence = np.sum(interface_distances == distance)
                occurence_list.append((occurence, distance))
            average_interface_distance = (np.sum([
                distance[0]*distance[1] for distance in occurence_list])
                / np.sum(np.array(occurence_list)[:, 0]))
            print('Average Off: ',
                  np.abs(average_interface_distance-average_distance))
            shift_preference.append(
                Shift(shift,
                      np.abs(average_interface_distance-average_distance)))
        shift_preference.sort(key=lambda x: x.average_off)
        best_shift = shift_preference[0]
        print('This is the best shift by averages: ', best_shift)
        supercell_1.translate_atoms(best_shift.shift)

    #Finally cut the cubes to the same size
    atoms_1 = supercell_1.get_atom_array_cartesian()
    min_atoms_1 = np.min(atoms_1, axis=0)[:2]
    max_atoms_1 = np.max(atoms_1, axis=0)[:2]
    atoms_2 = supercell_2.get_atom_array_cartesian()
    min_atoms_2 = np.min(atoms_2, axis=0)[:2]
    max_atoms_2 = np.max(atoms_2, axis=0)[:2]
    min_atoms = np.max(np.vstack((min_atoms_1, min_atoms_2)), axis=0)
    max_atoms = np.min(np.vstack((max_atoms_1, max_atoms_2)), axis=0)
    Cut = namedtuple('Cut', ['Point', 'Normal'])
    cut_list = [Cut([min_atoms[0], 0, 0], [-1, 0, 0]),
                Cut([max_atoms[0], 0, 0], [1, 0, 0]),
                Cut([0, min_atoms[1], 0], [0, -1, 0]),
                Cut([0, max_atoms[1], 0], [0, 1, 0])]
    gc.cut_grain(supercell_1, cut_list)
    gc.cut_grain(supercell_2, cut_list)
    translation = -np.array([min_atoms[0], min_atoms[1], 0])
    supercell_1.translate_atoms(translation)
    supercell_2.translate_atoms(translation)
    test_tool.xyz_file_output('supercell_1_finished', supercell_1.get_atom_array())
    test_tool.xyz_file_output('supercell_2_finished', supercell_2.get_atom_array())
    return (supercell_1, supercell_2)


def create_interface_simulation(supercell_1, supercell_2, interface_outline):
    '''
    Takes in two supercells that represent the slabs, uses these to create two
    lammps data files which it then integrates into a lammps input file.
    '''
    supercell_1.create_lammps_data_file_array(
        interface_outline.simulation_name+'_cell_1',
        interface_outline.potential_types)
    supercell_2.create_lammps_data_file_array(
        interface_outline.simulation_name+'_cell_2',
        interface_outline.potential_types)
