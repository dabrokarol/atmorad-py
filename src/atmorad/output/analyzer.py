import logging

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from atmorad.config import SimConfig

sns.set_theme(style="ticks", rc={"font.family": "serif"})


class ResultAnalyzer:
    def __init__(self, results_dict: dict, config: SimConfig):
        self.results_dict = results_dict
        self.config = config
        self.total_photons = config.engine.num_photons

    def experiment_summary(self) -> str:
        experiment_name = self.config.metadata.experiment_name
        num_photons = self.config.engine.num_photons

        total_time = self.results_dict.get("simulation_time_s", 0.0)
        cpu_time = self.results_dict.get("cpu_time_s", 0.0)

        reflected_toa = (self.results_dict.get("photons_escaped_toa", 0) / num_photons) * 100.0
        absorbed_surf = (self.results_dict.get("photons_absorbed_surface", 0) / num_photons) * 100.0
        absorbed_atm = (
            self.results_dict.get("photons_absorbed_atmosphere", 0) / num_photons
        ) * 100.0

        balance = reflected_toa + absorbed_surf + absorbed_atm

        return "\n".join(
            [
                f"\n---- Simulation Summary: {experiment_name} ----",
                f"Time: {total_time:.2f}s (Total) | {cpu_time:.2f}s (CPU)",
                f"Total Photons: {num_photons:_}\n",
                "Energy Distribution:",
                f"  {'Reflected (TOA)':<21}: {reflected_toa:>6.2f}%",
                f"  {'Surface Absorbed':<21}: {absorbed_surf:>6.2f}%",
                f"  {'Atmosphere Absorbed':<21}: {absorbed_atm:>6.2f}%",
                "  " + "-" * 30,
                f"  {'Energy Balance':<21}: {balance:>6.2f}%\n",
            ]
        )

    def plot_paths(self, title: str = "Sample 3D photon paths"):
        if "sample_paths" not in self.results_dict or not self.results_dict["sample_paths"]:
            return None

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        labeled_surface, labeled_above_toa, labeled_atmosphere = False, False, False

        Lx = self.config.environment.geometry.domain_size_x_km
        Ly = self.config.environment.geometry.domain_size_y_km
        limit_x, limit_y = Lx / 2, Ly / 2

        for path_id, path_coords in self.results_dict["sample_paths"].items():
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

            if self.results_dict["sample_absorbed_surface"][path_id]:
                color, alpha = "tab:green", 0.3
                lbl = "Absorbed by surface" if not labeled_surface else None
                labeled_surface = True
            elif self.results_dict["sample_escaped_toa"][path_id]:
                color, alpha = "tab:grey", 0.2
                lbl = "Escaped atmosphere" if not labeled_above_toa else None
                labeled_above_toa = True
            else:
                color, alpha = "tab:red", 0.3
                lbl = "Absorbed by atmosphere" if not labeled_atmosphere else None
                labeled_atmosphere = True

            ax.plot3D(X, Y, Z, alpha=alpha, color=color, label=lbl)

        limit_x = self.config.environment.geometry.domain_size_x_km / 2
        limit_y = self.config.environment.geometry.domain_size_y_km / 2

        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Pos x [km]")
        ax.set_ylabel("Pos y [km]")
        ax.set_zlabel("Pos z [km]")
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_y, limit_y)
        ax.set_zlim(0, self.results_dict["toa_z"])
        if labeled_surface or labeled_above_toa or labeled_atmosphere:
            ax.legend()
        return fig

    def plot_2d_map(self, flux_map: np.ndarray, title: str, label: str = "Normalized Flux"):
        x_edges = self.results_dict["x_edges"]
        y_edges = self.results_dict["y_edges"]

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
        flux_map = self.results_dict.get("surface_absorption_map_2d")

        if flux_map is None:
            logging.warning("Warning: No surface absorption map found in data.")
            return None

        return self.plot_2d_map(flux_map, title)

    def plot_toa_flux_map(self, title: str = "TOA Reflected Flux"):
        flux_map = self.results_dict.get("toa_flux_map_2d")

        if flux_map is None:
            logging.warning("Warning: No TOA flux map found in data.")
            return None

        return self.plot_2d_map(flux_map, title)

    def plot_flux_profile(self, title="Vertical Flux Profile"):
        if "flux_down" not in self.results_dict or "flux_up" not in self.results_dict:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))
        z = self.results_dict["measure_z"]
        flux_down = self.results_dict["flux_down"] / self.total_photons
        flux_up = self.results_dict["flux_up"] / self.total_photons
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
        if "absorption_profile_1d" not in self.results_dict:
            return None

        fig, ax = plt.subplots(figsize=(6, 8))
        boundaries = self.results_dict["measure_z"]
        profile = self.results_dict["absorption_profile_1d"] / self.total_photons
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
        if self.config.output.save_absorption_maps:
            fig_map = self.plot_surface_absorption_map()
            if fig_map:
                yield (fig_map, "surface_absorption_map.png")
                plt.close(fig_map)
            else:
                logging.warning("2d surface absorption map not generated")
            fig_toa_map = self.plot_toa_flux_map()
            if fig_toa_map:
                yield (fig_toa_map, "toa_flux_map.png")
                plt.close(fig_toa_map)
            else:
                logging.warning("2d toa flux map not generated")

        if self.config.output.save_incident_flux_maps:
            subfolder_name = "incident_flux"

            down_maps = self.results_dict.get("incident_flux_down_maps_2d", {})
            for z_val, flux_map in down_maps.items():
                title = f"Incident Downward Flux Map\nHeight: {z_val} km"
                fig = self.plot_2d_map(flux_map, title=title)
                if fig:
                    yield (fig, f"{subfolder_name}/downward_z_{z_val:g}km.png")
                    plt.close(fig)

            up_maps = self.results_dict.get("incident_flux_up_maps_2d", {})
            for z_val, flux_map in up_maps.items():
                title = f"Incident Upward Flux Map\nHeight: {z_val} km"
                fig = self.plot_2d_map(flux_map, title=title)
                if fig:
                    yield (fig, f"{subfolder_name}/upward_z_{z_val:g}km.png")
                    plt.close(fig)

        if self.config.output.save_vertical_profiles:
            fig_flux = self.plot_flux_profile()
            if fig_flux:
                yield (fig_flux, "vertical_flux_profile.png")
                plt.close(fig_flux)

            fig_heat = self.plot_absorption_profile()
            if fig_heat:
                yield (fig_heat, "absorption_profile.png")
                plt.close(fig_heat)

        if self.config.output.save_photon_paths:
            fig_paths = self.plot_paths()
            if fig_paths:
                yield (fig_paths, "3d_photon_paths.png")
                plt.close(fig_paths)
