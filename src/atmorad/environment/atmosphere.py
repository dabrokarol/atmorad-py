from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atmorad.constants import EPSILON, SAFE_INF, Z
from atmorad.models import PhotonBatch
from atmorad.physics import Scattering, rotate


@dataclass(slots=True)
class AtmosphericMedium:
    extinction_coeff: float
    ssa: float
    phase_function: Scattering

    def __post_init__(self):
        if self.extinction_coeff < 0:
            raise ValueError(
                f"Extinction coefficient should be non-negative, got {self.extinction_coeff}."
            )
        if not 0.0 <= self.ssa <= 1.0:
            raise ValueError(f"SSA should be in [0.0, 1.0], got {self.ssa}.")


class AtmosphericLayer:
    def __init__(
        self, thickness, components: Sequence[tuple[AtmosphericMedium, float]] | AtmosphericMedium
    ):
        if thickness < 0:
            raise ValueError(f"Thickness should be non-negative, got {thickness}.")

        self.thickness = thickness

        if isinstance(components, AtmosphericMedium):
            components = [(components, 1.0)]
        else:
            p_tot = sum(probability for _, probability in components)
            if not np.isclose(p_tot, 1, atol=EPSILON):
                raise ValueError(
                    f"Initialized layer probabilities don't sum to one, got {p_tot} Check atmospheric layers initialization."
                )

        self.components = components


class Atmosphere:
    def __init__(self, layers: Sequence[AtmosphericLayer]):
        boundaries = [0]
        unique_mediums = []
        max_layer_components = 0
        for layer in layers:
            boundaries.append(layer.thickness)
            for medium, probability in layer.components:
                if medium not in unique_mediums:
                    unique_mediums.append(medium)
            max_layer_components = max(max_layer_components, len(layer.components))

        layer_cdfs = np.zeros((len(layers), max_layer_components))
        layer_medium_ids = np.zeros((len(layers), max_layer_components)).astype(int)

        for i, layer in enumerate(layers):
            for j, (medium, probability) in enumerate(layer.components):
                if j == 0:
                    layer_cdfs[i, j] = probability
                else:
                    layer_cdfs[i, j] = (
                        layer_cdfs[i, j - 1] + probability
                    )  # cumulative summation to obtain cumulative density function
                layer_medium_ids[i, j] = unique_mediums.index(medium)
            width = len(layer.components)
            items_left = max_layer_components - width
            for j in range(width, width + items_left):
                layer_cdfs[i, j] = layer_cdfs[i, j - 1]
                layer_medium_ids[i, j] = layer_medium_ids[i, j - 1]

        self.ssas = np.array([medium.ssa for medium in unique_mediums])
        self.extinction_coeffs = np.array([medium.extinction_coeff for medium in unique_mediums])
        self.boundaries = np.cumsum(boundaries)

        self.phase_functions = [medium.phase_function for medium in unique_mediums]

        self.layer_cdfs = layer_cdfs
        self.layer_medium_ids = layer_medium_ids

    def _get_layer_idx(self, pos):
        layer_medium_idx = np.searchsorted(self.boundaries, pos[Z], "right") - 1
        layer_medium_idx = np.clip(
            layer_medium_idx, 0, len(self.boundaries) - 2
        )  # clip to valid indexes
        return layer_medium_idx

    def get_material_ids(self, pos, rand_1):
        layer_idx = self._get_layer_idx(pos)
        component_idx = np.argmax(rand_1[:, np.newaxis] < self.layer_cdfs[layer_idx], axis=1)
        return self.layer_medium_ids[
            layer_idx, component_idx
        ]  # array of column numbers, array of row numbers

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

    def process_scattering(
        self, batch: PhotonBatch, atmosphere_mask: np.ndarray, random_samples: np.ndarray
    ):
        to_scat = np.zeros_like(random_samples[0], dtype=bool)
        to_scat[atmosphere_mask] = self.is_scattered(
            batch.material_ids[atmosphere_mask], random_samples[0, atmosphere_mask]
        )
        cos_theta, sin_theta, cos_phi, sin_phi = self.scatter(
            batch.material_ids[to_scat], random_samples[1, to_scat], random_samples[2, to_scat]
        )
        batch.direction[:, to_scat] = rotate(
            batch.direction[:, to_scat], cos_theta, sin_theta, cos_phi, sin_phi
        )

        return batch, to_scat

    def distance_to_boundary(self, batch: PhotonBatch):
        layer_idx = self._get_layer_idx(batch.pos)

        delta_z = batch.pos[Z] - self.boundaries[layer_idx]
        travel_up = batch.direction[Z] > 0
        travel_down = batch.direction[Z] < 0
        travel_horizontal = batch.direction[Z] == 0

        lower_bound = self.boundaries[layer_idx]
        upper_bound = self.boundaries[layer_idx + 1]

        delta_z[travel_up] = upper_bound[travel_up] - batch.pos[Z, travel_up]
        delta_z[travel_down] = lower_bound[travel_down] - batch.pos[Z, travel_down]
        delta_z[travel_horizontal] = SAFE_INF

        return delta_z

    def tau_to_boundary(self, batch: PhotonBatch):
        delta_z = self.distance_to_boundary(batch)
        extinction_coeff = self.extinction_coeffs[batch.material_ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_to_boundary = np.abs(delta_z / batch.direction[Z]) * extinction_coeff
        return tau_to_boundary

    def tau_to_distance(self, batch: PhotonBatch, tau):
        extinction_coeff = self.extinction_coeffs[batch.material_ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = tau / extinction_coeff
        distance[extinction_coeff == 0] = SAFE_INF
        return distance

    def above_toa(self, pos):
        return pos[Z] > self.top_of_atmosphere

    def adjust_internal_boundaries(self, batch: PhotonBatch):
        escaped_toa = self.above_toa(batch.pos)
        batch.pos[:, escaped_toa] += (
            (self.top_of_atmosphere - batch.pos[Z, escaped_toa])
            / batch.direction[Z, escaped_toa]
            * batch.direction[:, escaped_toa]
        )
        batch.pos[Z, escaped_toa] = self.top_of_atmosphere + EPSILON

        for boundary_z in self.boundaries:
            on_boundary_mask = np.isclose(batch.pos[Z], boundary_z, atol=EPSILON)
            facing_up_mask = on_boundary_mask & (batch.direction[Z] > 0)
            facing_down_mask = on_boundary_mask & (batch.direction[Z] < 0)
            batch.pos[Z, facing_up_mask] += EPSILON
            batch.pos[Z, facing_down_mask] -= EPSILON
        return batch

    def get_spatial_indices(self, batch: PhotonBatch):
        return self._get_layer_idx(batch.pos)

    @property
    def top_of_atmosphere(self):
        return self.boundaries[-1]
