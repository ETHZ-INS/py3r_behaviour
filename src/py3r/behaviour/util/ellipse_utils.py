import numpy as np
from scipy.optimize import minimize

def ellipse_points(cx, cy, a, b, theta, n_points):
    """
    Generate n_points along the ellipse centered at (cx, cy) with axes a, b and rotation theta (radians).
    Returns a list of (x, y) tuples.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # Rotation
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.stack([x, y], axis=0)
    rot = R @ pts
    return [(cx + rot[0, i], cy + rot[1, i]) for i in range(n_points)]

def ellipse_residual(params, points, smallness_weight):
    xc, yc, a, b, theta = params
    # Penalize negative or too-small axes
    if a <= 0 or b <= 0:
        return 1e6
    residuals = []
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for x, y in points:
        xp = (x - xc) * cos_t + (y - yc) * sin_t
        yp = -(x - xc) * sin_t + (y - yc) * cos_t
        res = ((xp / a) ** 2 + (yp / b) ** 2) - 1
        residuals.append(res)
    fit_error = np.sum(np.square(residuals))
    size_penalty = smallness_weight * (a * b)
    return fit_error + size_penalty

def fit_ellipse_least_squares(points, smallness_weight=0.1):

    # Initial guess (center, axes, rotation)
    init = [np.mean(points[:,0]), np.mean(points[:,1]), 2, 1, 0]

    # Minimize the residuals
    result = minimize(ellipse_residual, init, args=(points, smallness_weight), method='Powell')

    if result.success:
        return result.x
    else:
        raise ValueError("Ellipse fitting failed.")
