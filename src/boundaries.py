import numpy as np

#### ROADMAP
# END GOAL:
# Surface can be a grid and each grid cell can have its own brdf function
# FOR NOW:
# Surface either absorbs all radiation (albedo=0) or acts as a mirror

def mirror_reflection(ori, r_theta, r_phi): #TODO: add possibility for functions to take parameters
    ori[2]*= -1 # mirror reflection
    return ori 
def no_reflection(ori, r_theta, r_phi):
    return 0

class Surface:
    def __init__(self, surf_config):
        
        self.albedo = surf_config['albedo']
        if self.albedo == 0:
            self.reflection_func = no_reflection
        elif surf_config['brdf_type'] == 'mirror':
            self.reflection_func = mirror_reflection
        else:
            raise KeyError(f"invalid brdf_type found in config: {surf_config['brdf_type']}")

        self.grid_density = surf_config['grid_density']
        self.dimensions = np.array(surf_config['dimensions']) * self.grid_density
        
        if surf_config['periodic'] == 'True':
            self.periodic = True
        else:
            self.periodic = False

        self.ground_id = np.zeros(shape=self.dimensions * self.grid_density) # for now all cells have the same ground type, to be changed later
        self.albedos = [self.albedo]
        self.brdfs = [self.reflection_func]

    def _snap_grid(self, pos):

        if self.periodic:
            snapped = np.floor(pos[:2] * self.grid_density).astype(int)
            return np.mod(snapped, self.dimensions[:, np.newaxis])
        else:
            raise FutureWarning('behaviour for non-periodic conditions is not yet implemented')
            # return np.floor(pos * self.grid_density)

    def check_reflection(self, pos, rand):
        
        cords = self._snap_grid(pos)

        if cords.size == 0:
            return np.zeros_like(rand, dtype=bool)
        
        ground_type = self.ground_id[cords[0], cords[1]]
        result_refl = np.zeros_like(rand)

        for i, albedo in enumerate(self.albedos):
            msk_i = (ground_type == i)
            result_refl[msk_i] = rand[msk_i] < albedo

        return result_refl
    
    def reflect(self, pos, ori, r_theta, r_phi): # todo: change to return cos_t, sin_t

        cords = self._snap_grid(pos)

        if cords.size == 0:
            return ori
        
        ground_type = self.ground_id[cords[0], cords[1]]
        result_ori = np.zeros_like(ori)

        for i, func in enumerate(self.brdfs):
            msk_i = ground_type == i
            result_ori[msk_i] = func(ori[:, msk_i], r_theta[msk_i], r_phi[msk_i])

        return result_ori

class Space:
    def __init__(self):
        pass