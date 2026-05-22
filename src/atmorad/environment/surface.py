from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atmorad.constants import EPSILON, X, Y, Z
from atmorad.models import PhotonBatch
from atmorad.physics import SurfaceReflection

from .surface_maps import BaseSurfaceMap


@dataclass(slots=True)
class SurfaceMaterial:
    albedo: float
    reflection: SurfaceReflection


class BaseSurface(ABC):
    """Abstract Base Class for all terrain types."""

    @abstractmethod
    def process_reflection(
        self, batch: PhotonBatch, surface_mask: np.ndarray, random_samples: np.ndarray
    ) -> tuple[PhotonBatch, np.ndarray]: ...

    @abstractmethod
    def is_below_ground(self, pos: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def adjust_surface_boundary(self, batch: PhotonBatch) -> PhotonBatch: ...

    @property
    @abstractmethod
    def domain_size(self) -> tuple[float, float]: ...


class FlatSurface(BaseSurface):
    def __init__(
        self,
        ground_map: BaseSurfaceMap,
        ground_types: Sequence[SurfaceMaterial],
        domain_x_km: float,
        domain_y_km: float,
        is_periodic: bool = True,
    ):
        self.ground_map = ground_map
        self.albedos = np.array([material.albedo for material in ground_types])
        self.reflections = [material.reflection for material in ground_types]

        self.domain_x = domain_x_km
        self.domain_y = domain_y_km
        self.is_periodic = is_periodic

    def process_reflection(
        self, batch: PhotonBatch, surface_mask: np.ndarray, random_samples: np.ndarray
    ):
        pos_hit = batch.pos[:, surface_mask]

        if self.is_periodic:
            pos_hit[X] = np.mod(pos_hit[X] + self.domain_x / 2, self.domain_x) - self.domain_x / 2
            pos_hit[Y] = np.mod(pos_hit[Y] + self.domain_y / 2, self.domain_y) - self.domain_y / 2

        material_ids = self.ground_map.get_material_ids(pos_hit)

        rand_albedo = random_samples[0, surface_mask]
        survived_albedo = rand_albedo < self.albedos[material_ids]

        to_reflect = np.zeros(batch.active_count, dtype=bool)
        to_reflect[surface_mask] = survived_albedo

        if np.any(survived_albedo):
            survivor_materials = material_ids[survived_albedo]
            survivor_dirs = batch.direction[:, to_reflect]
            survivor_r1 = random_samples[1, to_reflect]
            survivor_r2 = random_samples[2, to_reflect]

            for mat_id in np.unique(survivor_materials):
                material_mask = survivor_materials == mat_id
                survivor_dirs[:, material_mask] = self.reflections[mat_id].reflect(
                    survivor_dirs[:, material_mask],
                    survivor_r1[material_mask],
                    survivor_r2[material_mask],
                )

            batch.direction[:, to_reflect] = survivor_dirs

        return batch, to_reflect

    def is_below_ground(self, pos):
        return pos[Z] < 0

    def adjust_surface_boundary(self, batch: PhotonBatch):
        below_ground_mask = self.is_below_ground(batch.pos)
        batch.pos[:, below_ground_mask] += (
            (0 - batch.pos[Z, below_ground_mask])
            / batch.direction[Z, below_ground_mask]
            * batch.direction[:, below_ground_mask]
        )
        batch.pos[Z, below_ground_mask] = -EPSILON
        return batch

    @property
    def domain_size(self):
        return self.domain_x, self.domain_y
