from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atmorad.constants import GEOM_EPSILON, NUM_EPSILON, Z
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

        self.extinction_coeff = sum(
            medium.extinction_coeff * concentration for medium, concentration in components
        )
        self.ssa = (
            sum(
                medium.ssa * medium.extinction_coeff * concentration
                for medium, concentration in components
            )
            / self.extinction_coeff
        )
        self.components = components


class Atmosphere:
    def __init__(self, layers: Sequence[AtmosphericLayer]):
        boundaries = [0]
        unique_mediums = []
        max_layer_components = 0
        for layer in layers:
            boundaries.append(layer.thickness)
            for medium, concentration in layer.components:
                if medium not in unique_mediums:
                    unique_mediums.append(medium)
            max_layer_components = max(max_layer_components, len(layer.components))

        layer_cdfs = np.zeros((len(layers), max_layer_components))
        layer_medium_ids = np.zeros((len(layers), max_layer_components)).astype(int)

        for i, layer in enumerate(layers):
            for j, (medium, concentration) in enumerate(layer.components):
                probability = (concentration * medium.extinction_coeff) / layer.extinction_coeff
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

        self.ssas = np.array([layer.ssa for layer in layers])
        self.extinction_coeffs = np.array([layer.extinction_coeff for layer in layers])
        self.boundaries = np.cumsum(boundaries)

        self.layer_cdfs = layer_cdfs
        self.layer_medium_ids = layer_medium_ids
        self.phase_functions = [medium.phase_function for medium in unique_mediums]

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

    def scatter(self, medium_ids, rand_theta, rand_phi):
        cos_theta = np.zeros_like(rand_theta)
        sin_theta = np.zeros_like(rand_theta)
        cos_phi = np.zeros_like(rand_phi)
        sin_phi = np.zeros_like(rand_phi)

        for i, scat in enumerate(self.phase_functions):
            mask_i = medium_ids == i
            if np.any(mask_i):
                ct, st, cp, sp = scat(rand_theta[mask_i], rand_phi[mask_i])
                cos_theta[mask_i] = ct
                sin_theta[mask_i] = st
                cos_phi[mask_i] = cp
                sin_phi[mask_i] = sp

        return cos_theta, sin_theta, cos_phi, sin_phi

    def process_scattering(
        self, batch: PhotonBatch, scatter_mask: np.ndarray, rng: np.random.Generator
    ):
        ssas = self.get_ssas(batch.pos[:, scatter_mask])
        batch.weight[scatter_mask] *= ssas

        n_scat = np.count_nonzero(scatter_mask)

        cos_theta, sin_theta, cos_phi, sin_phi = self.scatter(
            batch.material_ids[scatter_mask],
            rng.random(n_scat),
            rng.random(n_scat),
        )

        batch.direction[:, scatter_mask] = rotate(
            batch.direction[:, scatter_mask], cos_theta, sin_theta, cos_phi, sin_phi
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
        delta_z[travel_horizontal] = np.inf

        return np.where(travel_horizontal, np.inf, delta_z / dir_z)

    def above_toa(self, pos):
        return pos[Z] > self.top_of_atmosphere

    def adjust_internal_boundaries(self, batch: PhotonBatch):
        escaped_toa = self.above_toa(batch.pos)

        if np.any(escaped_toa):
            dir_z = batch.direction[Z, escaped_toa]
            safe_mask = dir_z > NUM_EPSILON

            d = np.zeros_like(dir_z)
            d[safe_mask] = (
                (self.top_of_atmosphere + GEOM_EPSILON) - batch.pos[Z, escaped_toa][safe_mask]
            ) / dir_z[safe_mask]
            batch.pos[:, escaped_toa] += d * batch.direction[:, escaped_toa]

        # calculate distance to all boundaries
        diff = np.abs(batch.pos[Z, np.newaxis, :] - self.boundaries[:, np.newaxis])
        on_boundary_mask = np.any(diff <= GEOM_EPSILON, axis=0)

        if np.any(on_boundary_mask):
            dir_z = batch.direction[Z, on_boundary_mask]
            safe_mask = np.abs(dir_z) > NUM_EPSILON

            d = np.zeros_like(dir_z)
            d[safe_mask] = GEOM_EPSILON / np.abs(dir_z[safe_mask])
            batch.pos[:, on_boundary_mask] += batch.direction[:, on_boundary_mask] * d

        return batch

    def get_spatial_indices(self, batch: PhotonBatch):
        return self._get_layer_idx(batch.pos)

    def get_extinction_coeffs(self, pos: np.ndarray):
        return self.extinction_coeffs[self._get_layer_idx(pos)]

    def get_ssas(self, pos: np.ndarray):
        return self.ssas[self._get_layer_idx(pos)]

    @property
    def top_of_atmosphere(self):
        return self.boundaries[-1]
