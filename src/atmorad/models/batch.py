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

    @property
    def active_count(self):
        return self.is_active.sum()

    def deactivate_photons(self, mask):
        self.is_active[mask] = False

    def shrink_to_active(self):
        self.ids = self.ids[self.is_active]
        self.pos = self.pos[:, self.is_active]
        self.direction = self.direction[:, self.is_active]
        self.tau_to_travel = self.tau_to_travel[self.is_active]
        self.material_ids = self.material_ids[self.is_active]
        self.scatter_counts = self.scatter_counts[self.is_active]
        self.weight = self.weight[self.is_active]

        self.is_active = self.is_active[self.is_active]
