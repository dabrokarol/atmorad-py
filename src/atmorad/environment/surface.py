from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atmorad.constants import BOUNDARY_EPSILON, ZERO_TOLERANCE, X, Y, Z
from atmorad.physics import SurfaceReflection
from atmorad.physics.batch import PhotonBatch

from .surface_maps import BaseSurfaceMap


@dataclass(slots=True)
class SurfaceMaterial:
    albedo: float
    reflection: SurfaceReflection


class BaseSurface(ABC):
    """Abstract Base Class for all terrain types."""

    @abstractmethod
    def process_reflection(
        self, batch: PhotonBatch, surface_mask: np.ndarray, rng: np.random.Generator
    ) -> PhotonBatch: ...

    @abstractmethod
    def crossed_ground(self, pos: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def distance_to_surface(self, batch: PhotonBatch) -> np.ndarray: ...

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
        self, batch: PhotonBatch, surface_mask: np.ndarray, rng: np.random.Generator
    ):
        pos_hit = batch.pos[:, surface_mask]

        if self.is_periodic:
            pos_hit[X] = np.mod(pos_hit[X] + self.domain_x / 2, self.domain_x) - self.domain_x / 2
            pos_hit[Y] = np.mod(pos_hit[Y] + self.domain_y / 2, self.domain_y) - self.domain_y / 2

        material_ids = self.ground_map.get_material_ids(pos_hit)
        batch.weight[surface_mask] *= self.albedos[material_ids]

        hit_dirs = batch.direction[:, surface_mask]

        r1 = rng.random(np.count_nonzero(surface_mask))
        r2 = rng.random(np.count_nonzero(surface_mask))

        for mat_id in np.unique(material_ids):
            material_mask = material_ids == mat_id
            hit_dirs[:, material_mask] = self.reflections[mat_id].reflect(
                hit_dirs[:, material_mask],
                r1[material_mask],
                r2[material_mask],
            )

        batch.direction[:, surface_mask] = hit_dirs

        return batch

    def crossed_ground(self, pos):
        return pos[Z] <= 0

    def distance_to_surface(self, batch: PhotonBatch):
        return np.divide(
            -batch.pos[Z],
            batch.direction[Z],
            out=np.full(batch.active_count, np.inf),
            where=(batch.direction[Z] < -ZERO_TOLERANCE),
        )

    def adjust_surface_boundary(self, batch: PhotonBatch):
        below_ground_mask = self.crossed_ground(batch.pos)
        batch.pos[:, below_ground_mask] += (
            (0 - batch.pos[Z, below_ground_mask] + BOUNDARY_EPSILON)
            / batch.direction[Z, below_ground_mask]
            * batch.direction[:, below_ground_mask]
        )
        batch.pos[Z, below_ground_mask] = 0
        return batch

    @property
    def domain_size(self):
        return self.domain_x, self.domain_y
