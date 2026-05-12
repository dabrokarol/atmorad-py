import numpy as np

from dataclasses import dataclass
from typing import Sequence

from atmorad.physics.scattering import Scattering
  
@dataclass
class AtmosphericMedium:
    extinction_coeff: float
    ssa: float
    phase_function: Scattering
    def __post_init__(self):
        if self.extinction_coeff < 0:
            raise ValueError(f"Extinction coefficient should be non-negative, got {self.extinction_coeff}.")
        if not 0.0 <= self.ssa <= 1.0:
            raise ValueError(f"SSA should be in [0.0, 1.0], got {self.ssa}.")

class AtmosphericLayer:
    def __init__(self, thickness, components: Sequence[tuple[AtmosphericMedium, float]] | AtmosphericMedium):
        if thickness < 0:
            raise ValueError(f"Thickness should be non-negative, got {thickness}.")
        
        self.thickness = thickness

        if isinstance(components, AtmosphericMedium):
            components = [(components, 1.0)]
        else:
            p_tot = sum(probability for _, probability in components)
            if not np.isclose(p_tot, 1):
                raise ValueError(f"Initialized layer probabilities don't sum to one, got {p_tot} Check atmospheric layers initialization.")

        self.components = components

class Atmosphere:
    def __init__(self, layers: Sequence[AtmosphericLayer]):
        boundaries = [0]
        unique_mediums = []
        max_layer_components = 0
        for layer in layers:
            boundaries.append(layer.thickness)
            for medium, probability in layer.components:
                if not medium in unique_mediums:
                    unique_mediums.append(medium)
            max_layer_components = max(max_layer_components, len(layer.components))

        layer_cdfs = np.zeros((len(layers), max_layer_components))
        layer_medium_ids = np.zeros((len(layers), max_layer_components)).astype(int)

        for i, layer in enumerate(layers):
            for j, (medium, probability) in enumerate(layer.components):
                if j == 0:
                    layer_cdfs[i, j] = probability
                else:
                    layer_cdfs[i, j] = layer_cdfs[i, j-1] + probability # cumulative summation to obtain cumulative density function
                layer_medium_ids[i, j] = unique_mediums.index(medium)
            width = len(layer.components)
            items_left = max_layer_components - width
            for j in range(width, width + items_left):
                layer_cdfs[i, j] = layer_cdfs[i, j-1]
                layer_medium_ids[i, j] = layer_medium_ids[i, j-1]

        self.ssas = np.array([medium.ssa for medium in unique_mediums])
        self.extinction_coeffs = np.array([medium.extinction_coeff for medium in unique_mediums])
        self.boundaries = np.cumsum(boundaries)

        self.phase_functions = [medium.phase_function for medium in unique_mediums]

        self.layer_cdfs = layer_cdfs
        self.layer_medium_ids = layer_medium_ids

    def get_layer_idx(self, pos_z):
        layer_medium_idx = np.searchsorted(self.boundaries, pos_z, 'right') - 1
        layer_medium_idx = np.clip(layer_medium_idx, 0, len(self.boundaries) - 2) # clip to valid indexes
        return layer_medium_idx

    def get_mediums(self, pos, rand_1):
        layer_idx = self.get_layer_idx(pos[2])
        component_idx = np.argmax(rand_1[:, np.newaxis] < self.layer_cdfs[layer_idx], axis=1)
        return self.layer_medium_ids[layer_idx, component_idx] # array of column numbers, array of row numbers

    def is_scattered(self, medium_ids, rand_1):
        return rand_1 < self.ssas[medium_ids]
    
    def scatter(self, medium_ids, rand_theta, rand_phi):
        new_directions = np.zeros((4, rand_theta.size))
        for i, scat in enumerate(self.phase_functions):
            mask_i = medium_ids == i
            if np.any(mask_i):
                new_directions[:, mask_i] = scat(rand_theta[mask_i], rand_phi[mask_i])
        return new_directions
    
    def get_total_thickness(self):
        return self.boundaries[-1]
