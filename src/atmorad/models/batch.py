from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PhotonBatch:
    pos: np.ndarray
    direction: np.ndarray
    weight: np.ndarray
    tau_to_travel: np.ndarray
    is_active: np.ndarray
    ids: np.ndarray
    material_ids: np.ndarray
    scatter_counts: np.ndarray
    active_count: int = 0

    def __post_init__(self):
        self.active_count = self.ids.size

    @property
    def size(self):
        return self.ids.size

    def deactivate_photons(self, mask):
        newly_deactivated = self.is_active & mask
        self.is_active[mask] = False
        self.active_count -= np.count_nonzero(newly_deactivated)

    def shrink_to_active(self):
        self.ids = self.ids[self.is_active]
        self.pos = self.pos[:, self.is_active]
        self.direction = self.direction[:, self.is_active]
        self.tau_to_travel = self.tau_to_travel[self.is_active]
        self.material_ids = self.material_ids[self.is_active]
        self.scatter_counts = self.scatter_counts[self.is_active]
        self.weight = self.weight[self.is_active]

        self.is_active = self.is_active[self.is_active]
