import numpy as np

def orientation(theta, phi):
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

def rotate(ori, cos_t, sin_t, cos_p, sin_p):
    result = np.zeros_like(ori)
    big_z = np.abs(ori[2]) > 0.999
    small_z = ~big_z # inverted mask

    sqrt_z = np.sqrt(1 - ori[2, small_z]**2)

    result[:, small_z] = np.array((
        sin_t[small_z] / sqrt_z * (ori[0, small_z] * ori[2, small_z] * cos_p[small_z] - ori[1, small_z] * sin_p[small_z]) + ori[0, small_z] * cos_t[small_z],
        sin_t[small_z] / sqrt_z * (ori[1, small_z] * ori[2, small_z] * cos_p[small_z] + ori[0, small_z] * sin_p[small_z]) + ori[1, small_z] * cos_t[small_z],
        - sin_t[small_z] * cos_p[small_z] * sqrt_z + ori[2, small_z] * cos_t[small_z]
    ))
    result[:, big_z] = np.array((
        sin_t[big_z] * cos_p[big_z], sin_t[big_z] * sin_p[big_z], np.sign(ori[2, big_z]) * cos_t[big_z]
    ))
    
    return result

def henyey_greenstein(theta, g):
    return 0.5 * (1 - g**2 ) / (1 + g**2 - 2*g * np.cos(theta))**(3/2)

def hg_distribuant(theta, g):
    return (1 - g**2) / (2 * g) * (1 / (1 + g) - 1/np.sqrt(1 + g**2 - 2*g*np.cos(theta)))

def hg_cos_theta(r, g):
    if np.isclose(g, 0):
        return 2 * r - 1
    return 1 / (2*g) * (1 + g**2 - ((1 - g**2) / (2*g*r - g + 1))**2)