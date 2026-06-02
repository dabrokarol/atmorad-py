from typing import TYPE_CHECKING

import numpy as np

from atmorad.constants import ZERO_TOLERANCE, X, Y, Z
from atmorad.environment.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from atmorad.environment.surface import BaseSurface, FlatSurface, SurfaceMaterial
from atmorad.environment.surface_maps import SURFACE_MAPS
from atmorad.physics import sun_zenith_to_direction
from atmorad.physics.batch import PhotonBatch
from atmorad.physics.brdf import REFLECTION_MODELS
from atmorad.physics.phase_functions import SCATTERING_MODELS

if TYPE_CHECKING:
    from atmorad.config import EnvironmentConfig


class Scene:
    def __init__(self, surface: BaseSurface, atmosphere: Atmosphere) -> None:
        self.surface = surface
        self.atmosphere = atmosphere

    @classmethod
    def from_config(cls, env_config: "EnvironmentConfig") -> "Scene":

        atm_materials = {}
        for name, props in env_config.atmosphere_materials.items():
            scat_type = props.scattering["type"]
            scat_kwargs = {k: v for k, v in props.scattering.items() if k != "type"}
            phase_function = SCATTERING_MODELS[scat_type](**scat_kwargs)

            atm_materials[name] = AtmosphericMedium(
                extinction_coeff=props.extinction_coeff_per_km,
                ssa=props.ssa,
                phase_function=phase_function,
            )

        layers = []
        for layer_data in env_config.layers:
            components = [
                (atm_materials[comp.material], comp.concentration) for comp in layer_data.components
            ]
            layers.append(
                AtmosphericLayer(thickness=layer_data.thickness_km, components=components)
            )

        surf_materials = {}
        for name, mat_data in env_config.surface_materials.items():
            ref_type = mat_data.reflection["type"]
            ref_kwargs = {k: v for k, v in mat_data.reflection.items() if k != "type"}
            reflection_model = REFLECTION_MODELS[ref_type](**ref_kwargs)

            surf_materials[name] = SurfaceMaterial(
                albedo=mat_data.albedo, reflection=reflection_model
            )

        surf_cfg = env_config.surface
        map_name = surf_cfg["name"]

        map_data = SURFACE_MAPS[map_name]
        MapClass = map_data["class"]
        material_keys = map_data["material_keys"]

        material_names = [surf_cfg[key] for key in material_keys]
        ordered_materials = [surf_materials[name] for name in material_names]

        map_kwargs = {k: v for k, v in surf_cfg.items() if k not in material_keys and k != "name"}
        ground_map = MapClass(**map_kwargs)

        is_periodic = env_config.geometry.boundary_condition == "periodic"

        surface = FlatSurface(
            ground_map=ground_map,
            ground_types=ordered_materials,
            domain_x_km=env_config.geometry.domain_size_x_km,
            domain_y_km=env_config.geometry.domain_size_y_km,
            is_periodic=is_periodic,
        )
        return cls(surface=surface, atmosphere=Atmosphere(layers))

    def process_interactions(
        self,
        batch: PhotonBatch,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> PhotonBatch:
        """Scatters and reflects photons."""
        if np.any(scatter_mask):
            batch = self.atmosphere.process_scattering(batch, scatter_mask, rng)

        if np.any(surface_mask):
            batch = self.surface.process_reflection(batch, surface_mask, rng)

        return batch

    def above_toa(self, pos):
        return self.atmosphere.above_toa(pos)

    def at_surface(self, pos):
        return self.surface.crossed_ground(pos)

    def in_atmosphere(self, pos):
        return ~self.above_toa(pos) & ~self.at_surface(pos)

    def adjust_to_boundary_conditions(self, batch: PhotonBatch):
        batch = self.atmosphere.adjust_internal_boundaries(batch)
        batch = self.surface.adjust_surface_boundary(batch)
        return batch

    def start_pos(self, num_photons, rng: np.random.Generator):
        nx, ny = self.surface.domain_size
        pos = np.empty(shape=(3, num_photons), dtype=float)
        pos[X, :] = rng.uniform(-nx / 2, nx / 2, num_photons)
        pos[Y, :] = rng.uniform(-ny / 2, ny / 2, num_photons)
        pos[Z, :] = np.full(num_photons, self.atmosphere.top_of_atmosphere)
        return pos

    def start_direction(self, num_photons, theta_sun, phi_sun):
        theta = theta_sun / 180 * np.pi
        phi = phi_sun / 180 * np.pi
        direction = np.repeat(
            sun_zenith_to_direction(theta, phi)[:, np.newaxis], num_photons, axis=1
        )
        return direction

    def get_material_ids(self, pos, rng: np.random.Generator):
        rand_component = rng.uniform(0, 1, pos.shape[1])
        return self.atmosphere.get_material_ids(pos, rand_component)

    def move_photons(self, batch: PhotonBatch):
        ext_coeff = self.atmosphere.get_extinction_coeffs(batch.pos)
        dist_scatter = np.full(batch.active_count, np.inf)
        valid_ext = ext_coeff > ZERO_TOLERANCE

        dist_scatter[valid_ext] = batch.tau_to_travel[valid_ext] / ext_coeff[valid_ext]
        dist_boundary = self.atmosphere.distance_to_boundary(batch)
        dist_surface = self.surface.distance_to_surface(batch)
        dist_move = np.minimum.reduce([dist_scatter, dist_boundary, dist_surface])

        batch.pos += batch.direction * dist_move

        tau_consumed = dist_move * ext_coeff

        batch = self.adjust_to_boundary_conditions(batch)
        return batch, tau_consumed

    def get_final_photon_position_data(self, pos):
        return self.above_toa(pos), self.at_surface(pos), self.atmosphere.get_spatial_indices(pos)
