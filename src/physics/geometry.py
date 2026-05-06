import numpy as np

def orientation(cos_theta, sin_t, cos_phi, sin_p):
    return np.array((sin_t * cos_phi, sin_t * sin_p, cos_theta))

def rotate(ori, cos_theta, sin_t, cos_phi, sin_p):
    result = np.zeros_like(ori)
    big_z = np.abs(ori[2]) > 0.999
    small_z = ~big_z # inverted mask

    sqrt_z = np.sqrt(1 - ori[2, small_z]**2)

    result[:, small_z] = np.array((
        sin_t[small_z] / sqrt_z * (ori[0, small_z] * ori[2, small_z] * cos_phi[small_z] - ori[1, small_z] * sin_p[small_z]) + ori[0, small_z] * cos_theta[small_z],
        sin_t[small_z] / sqrt_z * (ori[1, small_z] * ori[2, small_z] * cos_phi[small_z] + ori[0, small_z] * sin_p[small_z]) + ori[1, small_z] * cos_theta[small_z],
        - sin_t[small_z] * cos_phi[small_z] * sqrt_z + ori[2, small_z] * cos_theta[small_z]
    ))
    result[:, big_z] = np.array((
        sin_t[big_z] * cos_phi[big_z], sin_t[big_z] * sin_p[big_z], np.sign(ori[2, big_z]) * cos_theta[big_z]
    ))
    
    return result