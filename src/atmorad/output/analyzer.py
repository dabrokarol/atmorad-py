import logging

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from atmorad.config import SimConfig

sns.set_theme(style="ticks", rc={"font.family": "serif"})


class ResultAnalyzer:
    def __init__(self, results_dict: dict, config: SimConfig):
        self.data = results_dict
        self.config = config
        self.total_photons = config.engine.num_photons

    def summary(self):
        summary_str = f"---- Simulation Summary ({self.config.metadata.experiment_name}) ----\n"
        summary_str += f"Total photons simulated: {self.total_photons}\n"

        if "simulation_time_s" in self.data:
            summary_str += f"Wall time: {self.data['simulation_time_s']:.2f} s\n"
        if "cpu_time_s" in self.data:
            summary_str += f"Total CPU time: {self.data['cpu_time_s']:.2f} s\n\n"

        reflected, absorbed_surf, absorbed_atm = 0.0, 0.0, 0.0

        num_photons = self.config.engine.num_photons
        reflected = self.data["photons_escaped_toa"] / num_photons
        absorbed_surf = self.data["photons_absorbed_surface"] / num_photons
        absorbed_atm = self.data["photons_absorbed_atmosphere"] / num_photons

        summary_str += f"Reflected (escaped toa): {reflected:.6f} ({reflected * 100:.2f}%)\n"
        summary_str += (
            f"Surface Absorption: {absorbed_surf:.6f} ({absorbed_surf * 100:.2f}%)\n"
        )
        summary_str += (
            f"Absorbed (absorbed by atmosphere): {absorbed_atm:.6f} ({absorbed_atm * 100:.2f}%)\n"
        )

        if reflected > 0 or absorbed_surf > 0 or absorbed_atm > 0:
            total_energy = reflected + absorbed_surf + absorbed_atm
            summary_str += "-----------------------------------\n"
            summary_str += f"Energy Balance Check: {total_energy:.6f} (Should be 1.0)\n"

        return summary_str

    def plot_paths(self, title: str = "Sample 3D photon paths"):
        if "sample_paths" not in self.data or not self.data["sample_paths"]:
            return None

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        labeled_surface, labeled_above_toa, labeled_atmosphere = False, False, False

        Lx = self.config.geometry.domain_size_x_km
        Ly = self.config.geometry.domain_size_y_km
        limit_x, limit_y = Lx / 2, Ly / 2

        for path_id, path_coords in self.data["sample_paths"].items():
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

            if self.data["sample_absorbed_surface"][path_id]:
                color, alpha = "tab:green", 0.3
                lbl = "Absorbed by surface" if not labeled_surface else None
                labeled_surface = True
            elif self.data["sample_escaped_toa"][path_id]:
                color, alpha = "tab:grey", 0.2
                lbl = "Escaped atmosphere" if not labeled_above_toa else None
                labeled_above_toa = True
            else:
                color, alpha = "tab:red", 0.3
                lbl = "Absorbed by atmosphere" if not labeled_atmosphere else None
                labeled_atmosphere = True

            ax.plot3D(X, Y, Z, alpha=alpha, color=color, label=lbl)

        limit_x = self.config.geometry.domain_size_x_km / 2
        limit_y = self.config.geometry.domain_size_y_km / 2

        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Pos x [km]")
        ax.set_ylabel("Pos y [km]")
        ax.set_zlabel("Pos z [km]")
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_y, limit_y)
        ax.set_zlim(0, self.data["toa_z"])
        if labeled_surface or labeled_above_toa or labeled_atmosphere:
            ax.legend()
        return fig

    def plot_2d_map(self, flux_map: np.ndarray, title: str, label: str = "Normalized Flux"):
        x_edges = self.data["x_edges"]
        y_edges = self.data["y_edges"]

        map_2d_norm = flux_map / self.total_photons

        fig, ax = plt.subplots(figsize=(8, 7))
        X, Y = np.meshgrid(x_edges, y_edges)

        mesh = ax.pcolormesh(X, Y, map_2d_norm.T, cmap=cmo.cm.solar, shading="flat")  # type: ignore
        ax.set_aspect("equal")
        fig.colorbar(
            mesh, ax=ax, label=label, orientation="horizontal", pad=0.1
        )
        ax.set_xlabel("Position X [km]")
        ax.set_ylabel("Position Y [km]")
        ax.set_title(title, fontsize=16)

        return fig

    def plot_surface_absorption_map(self, title: str = "Surface Absorption Map"):
        flux_map = self.data.get("surface_absorption_map_2d")

        if flux_map is None:
            logging.warning("Warning: No surface absorption map found in data.")
            return None

        return self.plot_2d_map(flux_map, title)

    def plot_toa_flux_map(self, title: str = "TOA Reflected Flux"):
        flux_map = self.data.get("toa_flux_map_2d")

        if flux_map is None:
            logging.warning("Warning: No TOA flux map found in data.")
            return None

        return self.plot_2d_map(flux_map, title)

    def plot_flux_profile(self, title="Vertical Flux Profile"):
        if "flux_down" not in self.data or "flux_up" not in self.data:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))
        z = self.data["measure_z"]
        flux_down = self.data["flux_down"] / self.total_photons
        flux_up = self.data["flux_up"] / self.total_photons
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
        if "absorption_profile_1d" not in self.data:
            return None

        fig, ax = plt.subplots(figsize=(6, 8))
        boundaries = self.data["layer_boundaries_z"]
        profile = self.data["absorption_profile_1d"] / self.total_photons
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
