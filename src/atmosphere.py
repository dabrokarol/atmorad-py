import numpy as np

from src.physics.scattering import Scattering
  
class AtmosphericMedium:
    def __init__(self, mu: float, albedo: float, scattering: Scattering):
        self.albedo = albedo
        self.mu = mu
        self.scattering = scattering

class AtmosphericLayer:
    def __init__(self, height, ingredients: list[tuple[AtmosphericMedium, float]]):
        self.height = height

        p = 0
        for i in ingredients:
            p += i[1]
        if not np.isclose(p, 1):
            raise ValueError("Initialized layer probabilities don't sum to one. Check atmospheric layers initialization.")

        self.ingredients = ingredients

class Atmosphere:
    def __init__(self, layers: list[AtmosphericLayer]):
        boundaries = [0]
        unique_mediums = []
        max_layer_width = 0
        for l in layers:
            boundaries.append(l.height)
            for medium, probability in l.ingredients:
                if not medium in unique_mediums:
                    unique_mediums.append(medium)
            max_layer_width = max(max_layer_width, len(l.ingredients))

        layer_cdfs = np.zeros((len(layers), max_layer_width))
        layer_medium_ids = np.zeros((len(layers), max_layer_width)).astype(int)

        for i, l in enumerate(layers):
            for j, (medium, probability) in enumerate(l.ingredients):
                if j == 0:
                    layer_cdfs[i, j] = probability
                else:
                    layer_cdfs[i, j] = layer_cdfs[i, j-1] + probability # cumulative summation to obtain cumulative density function
                layer_medium_ids[i, j] = unique_mediums.index(medium)
            width = len(l.ingredients)
            items_left = max_layer_width - width
            for j in range(width, width + items_left):
                layer_cdfs[i, j] = layer_cdfs[i, j-1]
                layer_medium_ids[i, j] = layer_medium_ids[i, j-1]

        self.albedos = np.array([medium.albedo for medium in unique_mediums])
        self.mus = np.array([medium.mu for medium in unique_mediums])
        self.boundaries = np.cumsum(boundaries)

        self.scatterings = [medium.scattering for medium in unique_mediums]

        self.layer_cdfs = layer_cdfs
        self.layer_medium_ids = layer_medium_ids

    def get_layer_idx(self, pos_z):
        layer_idx = np.searchsorted(self.boundaries, pos_z, 'right') - 1
        layer_idx = np.clip(layer_idx, 0, len(self.boundaries) - 2) # clip to valid indexes
        return layer_idx

    def get_mediums(self, pos, rand_1):
        layer_idx = self.get_layer_idx(pos[2])
        layer_medium_idx = np.argmax(rand_1[:, np.newaxis] < self.layer_cdfs[layer_idx], axis=1)
        return self.layer_medium_ids[layer_idx, layer_medium_idx] # array of column numbers, array of row numbers

    def check_scat(self, medium_ids, rand_1):
        return rand_1 < self.albedos[medium_ids]
    
    def scatter(self, medium_ids, rand_t, rand_p):
        scat_trig = np.zeros((4, rand_t.size))
        for i, scat in enumerate(self.scatterings):
            msk_i = medium_ids == i
            scat_trig[:, msk_i] = scat(rand_t[msk_i], rand_p[msk_i])
        return scat_trig
