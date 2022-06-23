import unittest
import time
import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import cProfile, pstats
import copy


class TestLinearAlgebra(unittest.TestCase):

    def test_rotation_matrix(self):
        '''
        Does rotation matrix produce the correct transform from a give axis and
        angle?
        '''
        angle = 20
        matrix = linalg.rotation_matrix('x', angle)
        expected_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(angle), -np.sin(angle)],
             [0, np.sin(angle), np.cos(angle)]])
        self.assertTrue(np.all(matrix == expected_matrix))

    def test_angle(self):
        '''
        Does it find the correct minimum angle between two vectors. Try various
        pairs to test this.
        '''
        vector1 = np.array([1, 1, 1])
        vector2 = np.array([0, 0, 1])
        angle = linalg.angle(vector1, vector2)
        expected_angle = np.arccos(1/np.sqrt(3))
        self.assertTrue(angle == expected_angle)
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 0, 1])
        angle = linalg.angle(vector1, vector2)
        expected_angle = np.pi/2
        self.assertTrue(angle == expected_angle)
        vector1 = np.array([1, 1, 1])
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = np.array([0, 0, 1])
        angle = linalg.angle(vector1, vector2)
        expected_angle = np.arccos(1/np.sqrt(3))
        self.assertTrue(angle == expected_angle)
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([1, 0, 0])
        angle = linalg.angle(vector1, vector2)
        expected_angle = 0
        self.assertTrue(angle == expected_angle)

    def test_vector_rotation_matrix(self):
        '''
        What effect does the angle and vector have on the rotation matrix? Does
        it return the expected rotation matrix for simple rotations?
        '''
        rotation_vector = np.array([0, 0, 1])
        angle = np.pi/4
        matrix = linalg.vector_rotation_matrix(angle, rotation_vector)
        vector = np.array([0, 0, 1])
        self.assertTrue(np.all(matrix @ vector == vector))
        vector = np.array([1, 0, 0])
        expected_vector = np.array([0.5, 0.5, 0])/np.linalg.norm([0.5, 0.5, 0])
        self.assertTrue(
            np.isclose(matrix @ vector, expected_vector, atol=0.0000001).all())
        #try opposite direction for rotation vector
        rotation_vector = np.array([0, 0, -1])
        matrix = linalg.vector_rotation_matrix(angle, rotation_vector)
        vector = np.array([0, 0, 1])
        self.assertTrue(np.all(matrix @ vector == vector))
        vector = np.array([1, 0, 0])
        expected_vector = (
            np.array([0.5, -0.5, 0])/np.linalg.norm([0.5, -0.5, 0]))
        self.assertTrue(
            np.isclose(matrix @ vector, expected_vector, atol=0.0000001).all())

    def test_rodrigues_rotation_matrix(self):
        '''
        Does the rodrigues formula return the expected rotation matrix for
        some simple rotations?
        '''
        rotation_vector = np.array([0, 0, 1])
        angle = 0.978972942
        rotation_matrix = (
            linalg.rodrigues_rotation_matrix(angle, rotation_vector))
        expected_matrix = linalg.rotation_matrix('z', angle)
        self.assertTrue(np.all(rotation_matrix == expected_matrix))
        rotation_vector = np.array([1, 1, 1])/np.sqrt(3)
        rotation_matrix = (
            linalg.rodrigues_rotation_matrix(angle, rotation_vector))
        comparison_matrix = (
            linalg.vector_rotation_matrix(angle, rotation_vector)
        )
        self.assertFalse(np.all(rotation_matrix == comparison_matrix))

    def test_rodrigues_rotaion_matrix_wont_accept_none_unit_vector(self):
        '''
        Does the function throw an error when asked to used a none unit vector?
        '''
        vector = np.array([0, 0, 1])
        vector = 1.0000001*vector
        angle = np.pi
        self.assertRaisesRegex(
            ValueError,
            f"Rotation Vector: {vector} is not a unit "
            "vector. Please input unit vector to use function")

    def test_normalise(self):
        '''
        Check that a given array will be normalised in the expected manner.
        Including a vector, or single row array.
        '''
        vector = np.array([1, 0, 2])
        vector = linalg.normalise(vector)
        expected_vector = np.array([1/np.sqrt(5), 0, 2/np.sqrt(5)])
        self.assertTrue(np.all(vector == expected_vector))
        self.assertTrue(np.isclose(np.linalg.norm(vector), 1, atol=1e-10))
        array = np.array(
            [[1, 0, 2], [3, 0, 0], [1, 5, 2], [-1, 2, 1], [-0.2, 1, 3.78]])
        array = linalg.normalise(array)
        expected_array = np.array([
            [1/np.sqrt(5), 0, 2/np.sqrt(5)],
            [1, 0, 0],
            [1/np.sqrt(30), 5/np.sqrt(30), 2/np.sqrt(30)],
            [-1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6)],
            [-0.2/np.sqrt(15.3284), 1/np.sqrt(15.3284), 3.78/np.sqrt(15.3284)]
        ])
        self.assertTrue(np.all(np.isclose(array, expected_array, atol=1e-10)))
        norms = np.linalg.norm(array, axis=1)
        expected_norms = np.array([1, 1, 1, 1, 1])
        self.assertTrue(np.all(np.isclose(norms, expected_norms, atol=1e-10)))

    def test_plane_normal(self):
        '''
        Does the function produce the correct plane normal using simple and
        complicated vector spaces.
        '''
        # Test for orthonormal vector space and simple intercepts.
        vector_space = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        intercepts = np.array([1, 1, 1])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([1, 1, 1])/np.sqrt(3)
        self.assertTrue(np.all(normal == expected_normal))

        # Test for orthonormal vector space and complex intercepts.
        intercepts = np.array([1, -2, 0.4])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([0.8, -0.4, 2])/np.sqrt(4.8)
        self.assertTrue(np.all(normal == expected_normal))

        # Test for orthogonal vector space and simple intercepts.
        vector_space = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
        intercepts = np.array([1, 1, 1])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([3, 3, 4])/np.sqrt(34)
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for orthogonal vector space and complex intercepts.
        vector_space = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 3]])
        intercepts = np.array([1, -1, 0])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([0, 0, -1])
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for complex vector space and simple intercepts
        vector_space = np.array([[4, 0, 0], [-2, 4, 0], [1, 1, 7]]).T
        intercepts = [1, 1, 1]
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([14, 21, 3])/np.sqrt(14**2+21**2+9)
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for complex vector space and complex intercepts
        intercepts = np.array([1, 0.1, -5])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([14, 147, -24.6])/np.sqrt(
            14**2+147**2+24.6**2)
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for orthogonal vector space and 2 infinite intercepts
        vector_space = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 4]])
        intercepts = np.array([1, np.inf, np.inf])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([1, 0, 0])
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for orthogonal vector space and 1 infinite intercept
        intercepts = np.array([1, np.inf, 1])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([2, 0, 1])/np.sqrt(5)
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for complex vector space and 2 infinite intercepts
        vector_space = np.array([[4, 0, 0], [-2, 4, 0], [1, 1, 7]]).T
        intercepts = np.array([np.inf, np.inf, 1])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([0, 0, 1])
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

        # Test for complex vector space 1 infinite intercept
        intercepts = np.array([np.inf, 1, 1])
        normal = linalg.plane_normal(vector_space, intercepts)
        expected_normal = np.array([0, 7, 3])/np.sqrt(58)
        self.assertTrue(np.all(np.isclose(normal, expected_normal)))

    def test_plane_points(self):
        '''
        Check the function gets plane points capable of creating a vector
        normal to the plane.
        '''
        # Check plane points rejects incorrect intercepts.
        vector_space = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        intercepts = [0, 0, 0]
        self.assertRaises(
            ValueError, linalg.plane_points, vector_space, intercepts)
        intercepts = [0, 0, 1]
        self.assertRaises(
            ValueError, linalg.plane_points, vector_space, intercepts)
        intercepts = [np.inf, np.inf, np.inf]
        self.assertRaisesRegex(
            ValueError,
            "Intercepts: \[inf inf inf\], cannot all be infinite, a maximum "
            "of two is permissable when defining a plane. Make sure your "
            "intercepts are defined correctly.",
            linalg.plane_points, vector_space, intercepts)
        intercepts = [1, 1, 1]
        expected_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(np.all(points == expected_points))

        # Check infinite intercepts are handled correctly.
        intercepts = [np.inf, 2, 1]
        expected_points = np.array([[1, 2, 0], [0, 2, 0], [0, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(np.all(points == expected_points))
        intercepts = [1, 1, np.inf]
        expected_points = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(
            np.all(np.isclose(points, expected_points, atol=1e-8)))
        intercepts = [1, np.inf, np.inf]
        expected_points = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(np.all(points == expected_points))

        # Check none integer intercepts work.
        intercepts = [0.5, np.inf, 1]
        expected_points = np.array([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(np.all(points == expected_points))

        # Check double infinite intercepts work
        intercepts = [np.inf, np.inf, 1]
        expected_points = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(np.all(points == expected_points))

        # Check none orthogonal vector spaces work.
        intercepts = [1, 1, np.inf]
        vector_space = np.array([[1, 0.2, 0], [0.1, 1, 0], [0, 0, 3]]).T
        expected_points = np.array([[1, 0.2, 0], [0.1, 1, 0], [1, 0.2, 3]])
        points = linalg.plane_points(vector_space, intercepts)
        self.assertTrue(
            np.all(np.isclose(points, expected_points, atol=1e-8)))

        # Check zero values work.
        intercepts = [1, 1, 0]
        expected_points = np.array([[1, 0.2, 0], [0.1, 1, 0], [0, 0, 0]])
        points = linalg.plane_points(vector_space, intercepts)

    def test_match_rows(self):
        '''
        Does match rows correctly compare the rows of an array against those in
        a match array - returning the correct boolean array?
        '''
        match_rows = np.array([[0, 1, 1, 1]])
        rows = np.array([[0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 1, 1]])
        matches = linalg.match_rows(rows, match_rows)
        expected_matches = [True, False, True]
        self.assertTrue(expected_matches == matches.tolist())

    def test_match_rows_multiple_rows(self):
        '''
        Does match rows correctly compare the rows of an array against those in
        a match array with multiple rows - returning the correct boolean array?
        '''
        match_rows = np.array([[0, 1, 1, 1], [0, 2, 2, 2]])
        rows = np.array([[0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 1, 1]])
        matches = linalg.match_rows(rows, match_rows)
        expected_matches = [True, False, True]
        self.assertTrue(expected_matches == matches.tolist())

    def test_match_rows_with_fuzziness(self):
        '''
        Does match rows correctly compare the rows of an array when fuzziness
        is added to the matching rows? Fuzziness is added in the form of an
        absolute tolerance.
        '''
        match_rows = np.array([[0, 1, 1, 1]])
        rows = np.array([[0, 0.9, 0.9, 1.1], [0, 1, 2, 2], [0, 1, 1, 1]])
        matches = linalg.match_rows(rows, match_rows, tolerance=0.1)
        expected_matches = [True, False, True]
        self.assertTrue(expected_matches == matches.tolist())

    def test_match_rows_with_fuzziness_two_rows(self):
        '''
        Does match rows correctly compare the rows of an array when fuzziness
        is added to the matching rows and more than one row is matched?
        Fuzziness is added as an absolute tolerance.
        '''
        match_rows = np.array([[0, 1, 1, 1], [0, 1.1, 1.9, 2.1]])
        rows = np.array([[0, 0.9, 0.9, 1.1], [0, 1, 2, 2], [0, 1, 1, 1]])
        matches = linalg.match_rows(rows, match_rows, tolerance=0.1)
        expected_matches = [True, True, True]
        self.assertTrue(expected_matches == matches.tolist())

    def test_angle_between_two_vectors(self):
        '''
        Check that the function works for two simple vectors.
        '''
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        angle = linalg.angle_between(vector1, vector2)
        expected_angle = np.pi/2
        self.assertTrue(angle == expected_angle)

    def test_angle_between_an_array_of_vectors_and_a_single_vector(self):
        '''
        Check that for an array of vectors and a single vector the correct
        values are returned.
        '''
        vectors1 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        vector2 = np.array([0, 1, 0])
        angles = linalg.angle_between(vectors1, vector2)
        expected_angles = np.array([np.pi/2, np.pi/2, 0, np.pi])
        self.assertTrue(np.all(angles == expected_angles))

    def test_angle_between_two_arrays_of_vectors(self):
        '''
        Check that for two arrays of vectors all the correct angles are found
        for each pair, and the order is convenient.
        '''
        vectors1 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        vectors2 = np.array([[0, 1, 0], [1, 0, 0]])
        angles = linalg.angle_between(vectors1, vectors2)
        expected_angles = np.array(
            [[np.pi/2, 0], [np.pi/2, np.pi], [0, np.pi/2], [np.pi, np.pi/2]])
        self.assertTrue(np.all(angles == expected_angles))

    def test_angle_between_two_complicated_arrays_of_vectors(self):
        '''
        Check that angle_between works for a complicated pair of vector arrays.
        '''
        vectors1 = np.array([[1, 0, 0], [-1, 1, 0], [-1, 1, 0], [0, -1, 0]])
        vectors2 = np.array([[0, 1, 0], [1, 0, 1]])
        angles = linalg.angle_between(vectors1, vectors2)
        expected_angles = np.array(
            [[np.pi/2, np.pi/4], [np.pi/4, (2/3)*np.pi],
             [np.pi/4, (2/3)*np.pi], [np.pi, np.pi/2]])
        self.assertTrue(
            np.all(np.isclose(angles, expected_angles, atol=0.001)))

    def test_group_by_angle(self):
        '''
        Does group by angle correctly a simple array of vectors?
        '''
        normals = np.array(
            [[1, 0, 0], [1.1, 0, 0], [0, 0.1, 0.1], [0, 0, 2], [0.1, 0.1, 1.9]]
            )
        expected_grouped_normals = [
            np.array([[1, 0, 0], [1.1, 0, 0]]),
            np.array([[0, 0.1, 0.1]]),
            np.array([[0, 0, 2], [0.1, 0.1, 1.9]])
            ]
        grouped_normals, index_groups = linalg.group_by_angle(normals, 0.08)
        for index in range(len(grouped_normals)):
            self.assertTrue(
                np.all(expected_grouped_normals[index] ==
                       grouped_normals[index]))
        expected_index_groups = [
            np.array([0, 1]), np.array([2]), np.array([3, 4])]
        for index in range(len(index_groups)):
            self.assertTrue(
                np.all(expected_index_groups[index] ==
                       index_groups[index]))
        grouped_normals, index_groups = linalg.group_by_angle(normals, np.pi)
        expected_grouped_normals = [
            np.array([[1, 0, 0], [1.1, 0, 0], [0, 0.1, 0.1],
            [0, 0, 2], [0.1, 0.1, 1.9]])]
        for index in range(len(grouped_normals)):
            self.assertTrue(
                np.all(expected_grouped_normals[index] ==
                       grouped_normals[index]))
        self.assertTrue(np.all(index_groups[0] == np.array([0, 1, 2, 3, 4])))

    def test_group_by_angle_complicated_vectors(self):
        '''
        Check group by angle correctly groups a more complciated group of
        vectors.
        '''
        array = np.array([
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [0.40824830532073975, 0.8164966106414795, 0.40824830532073975],
            [0.0, 0.0, -1.0], [0.0, -1.0, 0.0],
            [0.40824830532073975, 0.8164966106414795, 0.40824830532073975],
            [-1.0, 0.0, 0.0]])
        grouped_normals, indexes = linalg.group_by_angle(array, 0.01)
        expected_grouped_normals = [
            np.array([[-1., 0., 0.], [-1., 0., 0.]]),
            np.array([[0., -1., 0.], [0., -1., 0.], [0., -1., 0.],
                      [0., -1., 0.]]),
            np.array([[0.,  0., -1.], [0., 0., -1.]]),
            np.array([
                [0.40824830532073975, 0.8164966106414795, 0.40824830532073975],
                [0.40824830532073975, 0.8164966106414795, 0.40824830532073975]]
                )]
        for index in range(len(grouped_normals)):
            self.assertTrue(
                np.all(expected_grouped_normals[index] ==
                       grouped_normals[index]))


if __name__ == '__main__':
    current_directory = os.getcwd()
    package_directory_index = current_directory.index('grain_modeller')
    package_directory = current_directory[:package_directory_index+14]
    sys.path.append(package_directory+'/grain_modeller')
    import linear_algebra as linalg
    import testing_tools
    unittest.main()
