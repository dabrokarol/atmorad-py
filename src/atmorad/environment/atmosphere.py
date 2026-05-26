from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atmorad.constants import EPSILON, SAFE_INF, Z, SAFE_DIVISION
from atmorad.models import PhotonBatch
from atmorad.physics import Scattering, rotate


@dataclass(slots=True)
class AtmosphericMedium:
    extinction_coeff: float
    ssa: float
    phase_function: Scattering


class AtmosphericLayer:
    def __init__(
        self, thickness, components: Sequence[tuple[AtmosphericMedium, float]] | AtmosphericMedium
    ):
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

    def get_ssas(self, material_ids):
        return self.ssas[material_ids]

    def is_scattered(self, medium_ids, rand_1):
        return rand_1 < self.ssas[medium_ids]

    def scatter(self, medium_ids, rand_theta, rand_phi):
        new_directions = np.zeros((4, rand_theta.size))
        for i, scat in enumerate(self.phase_functions):
            mask_i = medium_ids == i
            if np.any(mask_i):
                new_directions[:, mask_i] = scat(rand_theta[mask_i], rand_phi[mask_i])
        return new_directions

    def process_scattering(
        self, batch: PhotonBatch, atmosphere_mask: np.ndarray, rng: np.random.Generator
    ):
        ssas = self.get_ssas(batch.material_ids[atmosphere_mask])
        batch.weight[atmosphere_mask] *= ssas

        n_scat = np.count_nonzero(atmosphere_mask)

        cos_theta, sin_theta, cos_phi, sin_phi = self.scatter(
            batch.material_ids[atmosphere_mask],
            rng.random(n_scat),
            rng.random(n_scat),
        )

        batch.direction[:, atmosphere_mask] = rotate(
            batch.direction[:, atmosphere_mask], cos_theta, sin_theta, cos_phi, sin_phi
        )

        return batch

    def distance_to_boundary(self, batch: PhotonBatch):
        layer_idx = self._get_layer_idx(batch.pos)

        pos_z = batch.pos[Z]
        dir_z = batch.direction[Z]

        delta_z = np.empty(batch.pos.shape[1], dtype=float)

        travel_up = dir_z > 0
        travel_down = dir_z < 0
        travel_horizontal = dir_z == 0

        delta_z[travel_up] = self.boundaries[layer_idx[travel_up] + 1] - pos_z[travel_up]
        delta_z[travel_down] = self.boundaries[layer_idx[travel_down]] - pos_z[travel_down]
        delta_z[travel_horizontal] = SAFE_INF

        return delta_z

    def step_to_boundary(self, batch: PhotonBatch):
        delta_z = self.distance_to_boundary(batch)

        dist_bound = np.abs(delta_z / batch.direction[Z])
        ext_coeff = self.extinction_coeffs[batch.material_ids]

        dist_scat = batch.tau_to_travel / ext_coeff
        dist_move = np.minimum(dist_bound, dist_scat)
        tau_consumed = dist_move * ext_coeff

        return dist_move, tau_consumed

    def above_toa(self, pos):
        return pos[Z] > self.top_of_atmosphere

    def adjust_internal_boundaries(self, batch: PhotonBatch):

        reflected_toa = self.above_toa(batch.pos)
        if np.any(reflected_toa):
            batch.pos[:, reflected_toa] += (
                (self.top_of_atmosphere - batch.pos[Z, reflected_toa])
                / batch.direction[Z, reflected_toa]
                * batch.direction[:, reflected_toa]
            )
            ds_toa = EPSILON / (np.abs(batch.direction[Z, reflected_toa]) + 1e-100)
            batch.pos[:, reflected_toa] += batch.direction[:, reflected_toa] * ds_toa

        diff = np.abs(batch.pos[Z, np.newaxis, :] - self.boundaries[:, np.newaxis])
        on_boundary_mask = np.any(diff <= EPSILON, axis=0)

        if np.any(on_boundary_mask):
            dir_z = batch.direction[Z, on_boundary_mask]
            ds = EPSILON / (np.abs(dir_z) + 1e-100)

            batch.pos[:, on_boundary_mask] += batch.direction[:, on_boundary_mask] * ds

        return batch

    def get_spatial_indices(self, batch: PhotonBatch):
        return self._get_layer_idx(batch.pos)

    @property
    def top_of_atmosphere(self):
        return self.boundaries[-1]
