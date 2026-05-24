from dataclasses import dataclass

import numpy as np

import atmorad
from atmorad import (
    BaseDetector,
    BaseResult,
    Scene,
    SimConfig,
    nc_attr,
    register_detector,
)


# 1. Define the result structure using AtmoRad's field wrappers
@dataclass(slots=True)
class FateResult(BaseResult):
    energy_absorbed_surface: float = nc_attr(normalize=True)
    energy_absorbed_atmosphere: float = nc_attr(normalize=True)
    energy_reflected_toa: float = nc_attr(normalize=True)


# 2. Implement the detector logic
@register_detector("fate", FateResult)
class FateDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.absorbed_surface = 0.0
        self.absorbed_atmosphere = 0.0
        self.reflected_toa = 0.0
        self.scene = scene

    def record_interaction(self, batch, old_direction, old_weight, scatter_mask, surface_mask):
        # Calculate deposited energy by subtracting the photon's new weight from its old weight.
        if np.any(scatter_mask):
            deposited = old_weight[scatter_mask] - batch.weight[scatter_mask]
            self.absorbed_atmosphere += np.sum(deposited)

        if np.any(surface_mask):
            deposited = old_weight[surface_mask] - batch.weight[surface_mask]
            self.absorbed_surface += np.sum(deposited)

    def record_termination(self, batch, terminated_mask):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        term_weight = batch.weight[terminated_mask]

        reflected_toa_mask = self.scene.above_toa(term_pos)
        if np.any(reflected_toa_mask):
            self.reflected_toa += np.sum(term_weight[reflected_toa_mask])

    def get_results(self) -> FateResult:
        return FateResult(
            energy_absorbed_surface=self.absorbed_surface,
            energy_absorbed_atmosphere=self.absorbed_atmosphere,
            energy_reflected_toa=self.reflected_toa,
        )


if __name__ == "__main__":
    # 3. Run the simulation
    results = atmorad.run("simulation.toml")
