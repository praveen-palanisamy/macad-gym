import numpy as np
import math


def _transform_matrix(transform):
    translation = transform.location
    rotation = transform.rotation

    # Transformation matrix
    cy = math.cos(np.radians(rotation.yaw))
    sy = math.sin(np.radians(rotation.yaw))
    cr = math.cos(np.radians(rotation.roll))
    sr = math.sin(np.radians(rotation.roll))
    cp = math.cos(np.radians(rotation.pitch))
    sp = math.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = translation.x
    matrix[1, 3] = translation.y
    matrix[2, 3] = translation.z
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr + sy * sr
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr

    return matrix


def transform_points(transform, points):
    """
        Given a 4x4 transformation matrix, transform an array of 3D points.
        Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
        """
    # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # get transform matrix
    matrix = _transform_matrix(transform)
    # Point transformation
    points = matrix * points
    # Return all but last row
    return points[0:3].transpose()
