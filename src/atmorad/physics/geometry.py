import numpy as np

from atmorad.constants import X, Y, Z

def orientation(cos_theta, sin_theta, cos_phi, sin_phi):
    return np.array((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta))

def sun_elevation_deg_to_direction(theta_sun, phi_sun):
    return sun_elevation_rad_to_direction(theta_sun*np.pi/180, phi_sun*np.pi/180)

def sun_elevation_rad_to_direction(theta_sun, phi_sun):
    cos_theta = np.cos(np.pi/2 - theta_sun)
    sin_theta = np.sin(np.pi/2 - theta_sun)
    cos_phi = np.cos(phi_sun + np.pi)
    sin_phi = np.sin(phi_sun + np.pi)
    return orientation(cos_theta, sin_theta, cos_phi, sin_phi)

def rotate(direction, cos_theta, sin_theta, cos_phi, sin_phi):
    result = np.zeros_like(direction)
    big_z = np.abs(direction[Z]) > 0.999
    small_z = ~big_z # inverted mask

    sqrt_z = np.sqrt(1 - direction[Z, small_z]**2)

    result[:, small_z] = np.array((
        sin_theta[small_z] / sqrt_z * (direction[X, small_z] * direction[Z, small_z] * cos_phi[small_z] - direction[Y, small_z] * sin_phi[small_z]) + direction[X, small_z] * cos_theta[small_z],
        sin_theta[small_z] / sqrt_z * (direction[Y, small_z] * direction[Z, small_z] * cos_phi[small_z] + direction[X, small_z] * sin_phi[small_z]) + direction[Y, small_z] * cos_theta[small_z],
        - sin_theta[small_z] * cos_phi[small_z] * sqrt_z + direction[Z, small_z] * cos_theta[small_z]
    ))
    result[:, big_z] = np.array((
        sin_theta[big_z] * cos_phi[big_z], sin_theta[big_z] * sin_phi[big_z], np.sign(direction[Z, big_z]) * cos_theta[big_z]
    ))
    
    return result