import logging

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from atmorad.config import SimConfig
from atmorad.models.results import SimulationResults

sns.set_theme(style="ticks", rc={"font.family": "serif"})


class ResultAnalyzer:
    def __init__(self, results: SimulationResults, config: SimConfig):
        self.results = results
        self.config = config
        self.total_photons = config.engine.num_photons
        self.detectors = self.results.detectors

    def experiment_summary(self) -> str:
        experiment_name = self.config.metadata.experiment_name
        num_photons = self.config.engine.num_photons

        total_time = self.results.engine.simulation_time_s
        cpu_time = self.results.engine.cpu_time_s

        fate_res = self.detectors.get("fate")

        if fate_res:
            escaped_toa = fate_res.photons_escaped_toa
            abs_surf = fate_res.photons_absorbed_surface
            abs_atm = fate_res.photons_absorbed_atmosphere
        else:
            escaped_toa = abs_surf = abs_atm = 0

        reflected_toa_pct = (escaped_toa / num_photons) * 100.0
        absorbed_surf_pct = (abs_surf / num_photons) * 100.0
        absorbed_atm_pct = (abs_atm / num_photons) * 100.0

        balance = reflected_toa_pct + absorbed_surf_pct + absorbed_atm_pct

        return "\n".join(
            [
                f"\n---- Simulation Summary: {experiment_name} ----",
                f"Time: {total_time:.2f}s (Total) | {cpu_time:.2f}s (CPU)",
                f"Total Photons: {num_photons:_}\n",
                "Energy Distribution:",
                f"  {'Reflected (TOA)':<21}: {reflected_toa_pct:>6.2f}%",
                f"  {'Surface Absorbed':<21}: {absorbed_surf_pct:>6.2f}%",
                f"  {'Atmosphere Absorbed':<21}: {absorbed_atm_pct:>6.2f}%",
                "  " + "-" * 30,
                f"  {'Energy Balance':<21}: {balance:>6.2f}%\n",
            ]
        )

    def plot_paths(self, title: str = "Sample 3D photon paths"):
        path_res = self.detectors.get("path_tracking")

        if not path_res or not path_res.sample_paths:
            return None

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        labeled_surface, labeled_above_toa, labeled_atmosphere = False, False, False

        Lx = self.config.environment.geometry.domain_size_x_km
        Ly = self.config.environment.geometry.domain_size_y_km
        limit_x, limit_y = Lx / 2, Ly / 2

        for path_id, path_coords in path_res.sample_paths.items():
            if not path_coords:
                continue

            coords = np.array(path_coords).T
            X, Y, Z = coords[0], coords[1], coords[2]

            X_wrapped = ((X + limit_x) % Lx) - limit_x
            Y_wrapped = ((Y + limit_y) % Ly) - limit_y

            jump_mask = (np.abs(np.diff(X_wrapped)) > limit_x) | (
                np.abs(np.diff(Y_wrapped)) > limit_y
            )
            jump_indices = np.where(jump_mask)[0] + 1

            X = np.insert(X_wrapped.astype(float), jump_indices, np.nan)
            Y = np.insert(Y_wrapped.astype(float), jump_indices, np.nan)
            Z = np.insert(Z.astype(float), jump_indices, np.nan)

            if path_res.sample_absorbed_surface.get(path_id):
                color, alpha = "tab:green", 0.3
                lbl = "Absorbed by surface" if not labeled_surface else None
                labeled_surface = True
            elif path_res.sample_escaped_toa.get(path_id):
                color, alpha = "tab:grey", 0.2
                lbl = "Escaped atmosphere" if not labeled_above_toa else None
                labeled_above_toa = True
            else:
                color, alpha = "tab:red", 0.3
                lbl = "Absorbed by atmosphere" if not labeled_atmosphere else None
                labeled_atmosphere = True

            ax.plot3D(X, Y, Z, alpha=alpha, color=color, label=lbl)

        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Pos x [km]")
        ax.set_ylabel("Pos y [km]")
        ax.set_zlabel("Pos z [km]")
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_y, limit_y)
        ax.set_zlim(0, path_res.toa_z)
        if labeled_surface or labeled_above_toa or labeled_atmosphere:
            ax.legend()
        return fig

    def plot_2d_map(
        self,
        flux_map: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        title: str,
        label: str = "Normalized Flux",
    ):
        map_2d_norm = flux_map / self.total_photons

        fig, ax = plt.subplots(figsize=(8, 7))
        X, Y = np.meshgrid(x_edges, y_edges)

        mesh = ax.pcolormesh(X, Y, map_2d_norm.T, cmap=cmo.cm.solar, shading="flat")  # type: ignore
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, label=label, orientation="horizontal", pad=0.1)
        ax.set_xlabel("Position X [km]")
        ax.set_ylabel("Position Y [km]")
        ax.set_title(title, fontsize=16)

        return fig

    def plot_surface_absorption_map(self, title: str = "Surface Absorption Map"):
        boundary_res = self.detectors.get("boundary_flux")
        if not boundary_res or boundary_res.surface_absorption_map_2d is None:
            logging.warning("Warning: No surface absorption map found in data.")
            return None

        return self.plot_2d_map(
            boundary_res.surface_absorption_map_2d,
            boundary_res.x_edges,
            boundary_res.y_edges,
            title,
        )

    def plot_toa_flux_map(self, title: str = "TOA Reflected Flux"):
        boundary_res = self.detectors.get("boundary_flux")

        if not boundary_res or boundary_res.toa_flux_map_2d is None:
            logging.warning("Warning: No TOA flux map found in data.")
            return None

        return self.plot_2d_map(
            boundary_res.toa_flux_map_2d, boundary_res.x_edges, boundary_res.y_edges, title
        )

    def plot_flux_profile(self, title="Vertical Flux Profile"):
        flux_res = self.detectors.get("vertical_flux")

        if not flux_res:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))
        z = flux_res.measure_z
        flux_down = flux_res.flux_down / self.total_photons
        flux_up = flux_res.flux_up / self.total_photons
        net_flux = flux_down - flux_up

        ax.plot(
            flux_down, z, label=r"Downward flux ($F^\downarrow$)", color="tab:blue", linewidth=2
        )
        ax.plot(flux_up, z, label=r"Upward flux ($F^\uparrow$)", color="tab:orange", linewidth=2)
        ax.plot(
            net_flux, z, label=r"Net flux ($F_{net}$)", color="black", linestyle="--", linewidth=2.5
        )

        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Normalized Flux", fontsize=12)
        ax.set_ylabel("Altitude Z [km]", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=11)
        ax.fill_betweenx(z, 0, net_flux, color="gray", alpha=0.1)

        fig.tight_layout()
        return fig

    def plot_absorption_profile(self, title="Atmospheric Absorption Profile"):
        abs_res = self.detectors.get("absorption_vertical")

        if not abs_res:
            return None

        fig, ax = plt.subplots(figsize=(6, 8))
        boundaries = abs_res.measure_z
        profile = abs_res.absorption_profile_1d / self.total_photons
        centers = (boundaries[:-1] + boundaries[1:]) / 2

        ax.barh(
            centers,
            profile,
            height=(boundaries[1:] - boundaries[:-1]),
            align="center",
            color="tab:red",
            alpha=0.6,
            edgecolor="black",
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Normalized Absorption", fontsize=12)
        ax.set_ylabel("Altitude Z [km]", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        return fig

    def generate_all_figures(self):
        if not self.config.output.save_plots:
            return

        boundary_res = self.detectors.get("boundary_flux")
        if boundary_res:
            boundary_plots = [
                (
                    self.plot_surface_absorption_map(),
                    "surface_absorption_map.png",
                    "2d surface absorption map not generated",
                ),
                (self.plot_toa_flux_map(), "toa_flux_map.png", "2d toa flux map not generated"),
            ]
            for fig, path, warning in boundary_plots:
                if fig:
                    yield (fig, path)
                    plt.close(fig)
                else:
                    logging.warning(warning)

        plane_res = self.detectors.get("plane_flux")
        if plane_res:
            map_configs = [
                ("Downward", plane_res.incident_flux_down_maps_2d, "downward"),
                ("Upward", plane_res.incident_flux_up_maps_2d, "upward"),
            ]
            for direction, flux_dict, prefix in map_configs:
                for z_val, flux_map in flux_dict.items():
                    title = f"Incident {direction} Flux Map\nHeight: {z_val} km"
                    fig = self.plot_2d_map(
                        flux_map, plane_res.x_edges, plane_res.y_edges, title=title
                    )
                    if fig:
                        yield (fig, f"incident_flux/{prefix}_z_{z_val:g}km.png")
                        plt.close(fig)

        flux_res = self.detectors.get("vertical_flux")
        if flux_res:
            fig_flux = self.plot_flux_profile()
            if fig_flux:
                yield (fig_flux, "vertical_flux_profile.png")
                plt.close(fig_flux)

        abs_res = self.detectors.get("absorption_vertical")
        if abs_res:
            fig_heat = self.plot_absorption_profile()
            if fig_heat:
                yield (fig_heat, "absorption_profile.png")
                plt.close(fig_heat)

        path_res = self.detectors.get("path_tracking")
        if path_res:
            fig_paths = self.plot_paths()
            if fig_paths:
                yield (fig_paths, "3d_photon_paths.png")
                plt.close(fig_paths)
