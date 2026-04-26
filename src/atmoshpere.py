import numpy as np

from src.physics import hg_cos_theta

# TODO: allow for non uniform atmosphere, add layers with variable mu

class Atmosphere:
    def __init__(self, atm_config):
        self.g = atm_config['scattering']['g'] 
        scat_type = atm_config['scattering']['type']
        if scat_type == 'henyey-greenstein':
            self.scat_func = hg_cos_theta
        else:
            raise KeyError(f"Unknown scattering func {scat_type}, check config")
        
        self.ss_albedo = atm_config['omega']
        

        self.tau_star = atm_config['tau_star']
        self.height = atm_config['height']
        self.mu = self.tau_star / self.height

    def calc_dist(self, pos, ori, tau):
        return tau / self.height # uniform atmosphere for now
    
    def check_scat(self, pos, ori, rand):
        return rand < self.ss_albedo # uniform albedo for now
    
    def scatter(self, pos, ori, rand_t, rand_p):
        cos_t = hg_cos_theta(rand_t, self.g)
        cos_p = np.cos(2*np.pi * rand_p)

        return cos_t, cos_p
    
    def check_reached_space(self, pos, ori):
        reached = pos[2] < 0
        pos[:, reached] += (0 - pos[2, reached]) / ori[2, reached] * ori[:, reached]
        return pos, reached
    
    def check_reached_surf(self, pos, ori):
        reached = pos[2] > self.height
        pos[:, reached] += (self.tau_star - pos[2, reached]) / ori[2, reached] * ori[:, reached]
        return pos, reached