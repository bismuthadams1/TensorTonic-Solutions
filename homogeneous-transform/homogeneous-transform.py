import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points= np.asarray(points)
    T = np.asarray(T)
    if len(points.shape) > 1:
        r, c = points.shape
    else:
        points = points[None, :]
        r = 1
    # points = np.append(points, 1)[np.newaxis, ...]
    one_dim = np.ones(r).reshape(-1,1)

    points = np.concatenate([points, one_dim], axis=1)

    out = points @ np.transpose(T)

    out = np.delete(out, obj= -1, axis = 1)

    return np.squeeze(out)