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
        self.detectors = self.results.detector_results

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
            escaped_toa = abs_surf = abs_atm = 0.0

        reflected_toa_pct = (escaped_toa / num_photons) * 100.0
        absorbed_surf_pct = (abs_surf / num_photons) * 100.0
        absorbed_atm_pct = (abs_atm / num_photons) * 100.0

        balance = reflected_toa_pct + absorbed_surf_pct + absorbed_atm_pct

        return "\n".join(
            [
                f"\n---- Simulation Summary: {experiment_name} ----",
                f"Time: {total_time:.2f}s (Total) | {cpu_time:.2f}s (CPU)",
                f"Total Photons / Energy: {num_photons:_}\n",
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

        if not path_res or len(path_res.sample_paths_3d) == 0:
            return None

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        labeled_surface, labeled_above_toa, labeled_atmosphere = False, False, False

        Lx = self.config.environment.geometry.domain_size_x_km
        Ly = self.config.environment.geometry.domain_size_y_km
        limit_x, limit_y = Lx / 2, Ly / 2

        paths = path_res.sample_paths_3d
        num_paths = paths.shape[0]

        for i in range(num_paths):
            X = paths[i, :, 0]
            Y = paths[i, :, 1]
            Z = paths[i, :, 2]

            with np.errstate(invalid="ignore"):
                X_wrapped = ((X + limit_x) % Lx) - limit_x
                Y_wrapped = ((Y + limit_y) % Ly) - limit_y

                jump_mask = (np.abs(np.diff(X_wrapped)) > limit_x) | (
                    np.abs(np.diff(Y_wrapped)) > limit_y
                )
                jump_indices = np.where(jump_mask)[0] + 1

            X_plot = np.insert(X_wrapped.astype(float), jump_indices, np.nan)
            Y_plot = np.insert(Y_wrapped.astype(float), jump_indices, np.nan)
            Z_plot = np.insert(Z.astype(float), jump_indices, np.nan)

            if path_res.sample_absorbed_surface[i]:
                color, alpha = "tab:green", 0.3
                lbl = "Absorbed by surface" if not labeled_surface else None
                labeled_surface = True
            elif path_res.sample_escaped_toa[i]:
                color, alpha = "tab:grey", 0.2
                lbl = "Escaped atmosphere" if not labeled_above_toa else None
                labeled_above_toa = True
            else:
                color, alpha = "tab:red", 0.3
                lbl = "Absorbed by atmosphere" if not labeled_atmosphere else None
                labeled_atmosphere = True

            ax.plot3D(X_plot, Y_plot, Z_plot, alpha=alpha, color=color, label=lbl)

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
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        title: str,
        label: str = "Normalized Flux",
    ):
        map_2d_norm = flux_map / self.total_photons

        fig, ax = plt.subplots(figsize=(8, 7))
        X, Y = np.meshgrid(x_centers, y_centers)

        mesh = ax.pcolormesh(X, Y, map_2d_norm.T, cmap=cmo.cm.solar, shading="nearest")
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, label=label, orientation="horizontal", pad=0.1)
        ax.set_xlabel("Position X [km]")
        ax.set_ylabel("Position Y [km]")
        ax.set_title(title, fontsize=16)

        return fig

    def plot_surface_absorption_map(self, title: str = "Surface Absorption Map"):
        surf_res = self.detectors.get("surface_absorption")
        if not surf_res or surf_res.surface_absorption_map_2d is None:
            logging.warning("Warning: No surface absorption map found in data.")
            return None

        return self.plot_2d_map(
            surf_res.surface_absorption_map_2d,
            surf_res.x_centers,
            surf_res.y_centers,
            title,
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

        z_centers = abs_res.z_centers
        profile = abs_res.absorption_profile_1d / self.total_photons
        spacing = self.config.detectors.vertical_profiles_resolution_km

        ax.barh(
            z_centers,
            profile,
            height=spacing,
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

        surf_res = self.detectors.get("surface_absorption")
        if surf_res:
            fig = self.plot_surface_absorption_map()
            if fig:
                yield (fig, "surface_absorption_map.png")
                plt.close(fig)
            else:
                logging.warning("2d surface absorption map not generated")

        plane_res = self.detectors.get("plane_flux")
        if plane_res:
            for i, z_val in enumerate(plane_res.measure_z):
                title_down = f"Incident Downward Flux Map\nHeight: {z_val} km"
                fig_down = self.plot_2d_map(
                    plane_res.incident_flux_down_3d[i],
                    plane_res.x_centers,
                    plane_res.y_centers,
                    title=title_down,
                )
                if fig_down:
                    yield (fig_down, f"incident_flux/downward_z_{z_val:g}km.png")
                    plt.close(fig_down)

                title_up = f"Incident Upward Flux Map\nHeight: {z_val} km"
                fig_up = self.plot_2d_map(
                    plane_res.incident_flux_up_3d[i],
                    plane_res.x_centers,
                    plane_res.y_centers,
                    title=title_up,
                )
                if fig_up:
                    yield (fig_up, f"incident_flux/upward_z_{z_val:g}km.png")
                    plt.close(fig_up)

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
