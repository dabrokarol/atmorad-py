import numpy as np

def orientation(cos_theta, sin_theta, cos_phi, sin_phi):
    return np.array((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta))

def rotate(direction, cos_theta, sin_theta, cos_phi, sin_phi):
    result = np.zeros_like(direction)
    big_z = np.abs(direction[2]) > 0.999
    small_z = ~big_z # inverted mask

    sqrt_z = np.sqrt(1 - direction[2, small_z]**2)

    result[:, small_z] = np.array((
        sin_theta[small_z] / sqrt_z * (direction[0, small_z] * direction[2, small_z] * cos_phi[small_z] - direction[1, small_z] * sin_phi[small_z]) + direction[0, small_z] * cos_theta[small_z],
        sin_theta[small_z] / sqrt_z * (direction[1, small_z] * direction[2, small_z] * cos_phi[small_z] + direction[0, small_z] * sin_phi[small_z]) + direction[1, small_z] * cos_theta[small_z],
        - sin_theta[small_z] * cos_phi[small_z] * sqrt_z + direction[2, small_z] * cos_theta[small_z]
    ))
    result[:, big_z] = np.array((
        sin_theta[big_z] * cos_phi[big_z], sin_theta[big_z] * sin_phi[big_z], np.sign(direction[2, big_z]) * cos_theta[big_z]
    ))
    
    return result