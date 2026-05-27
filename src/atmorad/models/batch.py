from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PhotonBatch:
    pos: np.ndarray
    direction: np.ndarray
    weight: np.ndarray
    tau_to_travel: np.ndarray
    is_active: np.ndarray
    ids: np.ndarray
    scatter_counts: np.ndarray
    active_count: int = 0

    old_pos: np.ndarray = field(init=False)
    old_direction: np.ndarray = field(init=False)
    old_weight: np.ndarray = field(init=False)

    def __post_init__(self):
        self.active_count = int(self.ids.size)
        self.old_pos = np.empty_like(self.pos)
        self.old_direction = np.empty_like(self.direction)
        self.old_weight = np.empty_like(self.weight)

    @property
    def size(self):
        return self.ids.size

    def deactivate_photons(self, dead_mask: np.ndarray):
        newly_deactivated = self.is_active & dead_mask
        self.is_active[dead_mask] = False
        self.active_count -= int(np.count_nonzero(newly_deactivated))

    def shrink_to_active(self):
        mask = self.is_active

        self.ids = self.ids[mask]
        self.pos = self.pos[:, mask]
        self.old_pos = self.old_pos[:, mask]
        self.direction = self.direction[:, mask]
        self.old_direction = self.old_direction[:, mask]
        self.tau_to_travel = self.tau_to_travel[mask]
        self.scatter_counts = self.scatter_counts[mask]
        self.weight = self.weight[mask]
        self.old_weight = self.old_weight[mask]

        self.is_active = np.ones(self.active_count, dtype=bool)

        return mask.size - np.count_nonzero(mask)

    def update_old_state(self):
        np.copyto(self.old_pos, self.pos)
        np.copyto(self.old_direction, self.direction)
        np.copyto(self.old_weight, self.weight)
