import numpy as np
from scipy.spatial.distance import cdist

def demean_points(points):
    dist = cdist(points, points)
    mean_dist = np.mean(dist, axis=0)
    mean_point = points[np.argmin(mean_dist)]
    return points - mean_point, mean_point

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def reflection_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, s), (s, -c)))
    
def rotation(points, theta, origin = None):
    if origin is None:
        _points, origin = demean_points(points)
    else:
        _points = points - origin
    R = rotation_matrix(theta)
    rotated_points = np.dot(_points, R)
    return rotated_points + origin

def reflection(points, theta, origin = None):
    if origin is None:
        _points, origin = demean_points(points)
    else:
        _points = points - origin
    S = reflection_matrix(theta)
    reflected_points = np.dot(_points, S)
    return reflected_points + origin

def translation(points, translation_vector):
    return points + translation_vector

def scale(points, scale_constant):
    demeaned_points, mean_point = demean_points(points)
    return demeaned_points * scale_constant + mean_point


if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--input", help="The file where the input points are stored", type=str)
    parser.add_argument("--output", help="The file where the output points going to be stored", type=str)
    parser.add_argument("--scale", default=1, help="Scale of points", type=float)
    parser.add_argument("--translationX", default=0, help="X value by which to translate the data", type = int)
    parser.add_argument("--translationY", default=0, help="Y value by which to translate the data", type = int)
    parser.add_argument("--rotationAngle", default=0, help="Angle by which to rotate data", type=int)
    parser.add_argument("--rotationOriginX", default=None, help="X coordinate of the origin around which coordinates will be rotated", type=int)
    parser.add_argument("--rotationOriginY", default=None, help="Y coordinate of the origin around which coordinate will be rotated", type=int)
    parser.add_argument("--doReflection", action="store_true", help="Whether to perform a reflection")
    parser.add_argument("--reflectionAngle", default=0, help="Angle that the reflection axis lies on", type=int)
    parser.add_argument("--reflectionOriginX", default=None, help="X value of the origin of the reflection axis", type=int)
    parser.add_argument("--reflectionOriginY", default=None)
    args = parser.parse_args()
    
    # Load the input data
    points = np.load(args.input)
    
    # Perform reflection
    if args.doReflection:
        if args.reflectionOriginX is None or args.reflectionOriginY is None:
            reflectionOrigin = None
        else:
            reflectionOrigin = [args.reflectionOriginX, args.reflectionOriginY]
        points = reflection(points, np.radians(args.reflectionAngle), reflectionOrigin)

    # Perform rotation
    if args.rotationOriginX is None or args.rotationOriginY is None:
        rotationOrigin = None
    else:
        rotationOrigin = [args.rotationOriginX, args.rotationOriginY]
    points = rotation(points, np.radians(args.rotationAngle), rotationOrigin)
    
    # Perform translation
    translation_vector = [args.translationX, args.translationY]
    points = translation(points, translation_vector)
    
    # Perform scaling
    points = scale(points, args.scale)
    
    # Convert to type np.int
    points = points.astype(np.int)
    
    # Send output to file
    np.save(args.output, points)