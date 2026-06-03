import json
import logging

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_theme(
    style="ticks",
    rc={
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.titlepad": 15,
        "axes.titleweight": "500",
        "axes.labelsize": 17,
        "axes.labelpad": 10,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "legend.frameon": False,
        "figure.dpi": 100,
    },
)


class ResultAnalyzer:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.det_types = {}
        try:
            config_str = str(ds.attrs.get("_simulation_config", "{}"))
            config = json.loads(config_str)

            active_detectors = config.get("detectors", {}).get("active", [])

            if not active_detectors:
                logging.warning("No active detectors found in the dataset configuration.")

            if isinstance(active_detectors, list):
                self.det_types = {det_name: det_name for det_name in active_detectors}
            elif isinstance(active_detectors, dict):
                self.det_types = {k: k for k in active_detectors.keys()}

        except json.JSONDecodeError:
            logging.error("Could not decode _simulation_config from dataset attributes.")

    def _get_scalar(self, name: str, default: float = 0.0) -> float:
        if name in self.ds:
            return float(self.ds[name].values)
        if name in self.ds.attrs:
            return float(self.ds.attrs[name])
        return default

    def experiment_summary(self) -> str:
        experiment_name = self.ds.attrs.get("experiment_name", "")
        scenario_name = self.ds.attrs.get("scenario_name", "")

        total_time = self.ds.attrs.get("simulation_time_s", 0.0)
        total_photons = int(self.ds.attrs.get("num_photons", 0))

        outgoing_toa = self._get_scalar("energy_toa_outgoing")
        abs_surf = self._get_scalar("energy_surface_absorbed")
        abs_atm = self._get_scalar("energy_atmosphere_absorbed")

        outgoing_toa_pct = outgoing_toa * 100.0
        absorbed_surf_pct = abs_surf * 100.0
        absorbed_atm_pct = abs_atm * 100.0

        conservation = outgoing_toa_pct + absorbed_surf_pct + absorbed_atm_pct

        return "\n".join(
            [
                "Simulation complete",
                "",
                f"experiment: {experiment_name}",
                f"scenario: {scenario_name}",
                "",
                f"runtime: {total_time:.2f} s",
                f"photons: {total_photons:_}",
                "",
                "Energy distribution",
                "-" * 30,
                f"{'toa escape':<24}{outgoing_toa_pct:>6.2f}%",
                f"{'surface absorption':<24}{absorbed_surf_pct:>6.2f}%",
                f"{'atmospheric absorption':<24}{absorbed_atm_pct:>6.2f}%",
                "-" * 30,
                f"{'energy conservation':<24}{conservation:>6.2f}%",
                "",
            ]
        )

    def _infer_domain_size(self, paths_array: np.ndarray) -> tuple[float, float]:
        Lx = self.ds.attrs.get("domain_size_x_km")
        Ly = self.ds.attrs.get("domain_size_y_km")

        if Lx is None or Ly is None:
            if "x_surface" in self.ds.coords and Lx is None:
                x_vals = self.ds.coords["x_surface"].values
                if len(x_vals) > 1:
                    Lx = float(x_vals.max() - x_vals.min() + (x_vals[1] - x_vals[0]))

            if "y_surface" in self.ds.coords and Ly is None:
                y_vals = self.ds.coords["y_surface"].values
                if len(y_vals) > 1:
                    Ly = float(y_vals.max() - y_vals.min() + (y_vals[1] - y_vals[0]))

        if Lx is None:
            Lx = float(np.nanmax(np.abs(paths_array[:, :, 0]))) * 2.0
        if Ly is None:
            Ly = float(np.nanmax(np.abs(paths_array[:, :, 1]))) * 2.0

        return max(Lx, 1.0), max(Ly, 1.0)

    def plot_paths(self, title: str = "Sample 3D photon paths", max_paths: int = 500):
        if "paths" not in self.ds:
            return None

        paths = self.ds["paths"].values
        num_paths = paths.shape[0]

        if num_paths == 0:
            return None

        if num_paths > max_paths:
            logging.warning(
                f"Limiting 3D plot to {max_paths} paths (out of {num_paths}) to prevent freezing."
            )
            paths = paths[:max_paths]
            num_paths = max_paths

        absorbed_surface = self.ds["absorbed_surface"].values[:num_paths]
        outgoing_toa = self.ds["escaped_toa"].values[:num_paths]

        toa_z = self.ds.attrs.get("toa_z_km", 10.0)

        Lx, Ly = self._infer_domain_size(paths)
        limit_x, limit_y = Lx / 2.0, Ly / 2.0

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        idx_surf = np.where(absorbed_surface)[0]
        idx_toa = np.where(outgoing_toa)[0]
        idx_atm = np.where(~absorbed_surface & ~outgoing_toa)[0]

        def plot_group(indices, color, alpha, label):
            if len(indices) == 0:
                return

            all_x, all_y, all_z = [], [], []
            for i in indices:
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

                all_x.extend(X_plot.tolist() + [np.nan])
                all_y.extend(Y_plot.tolist() + [np.nan])
                all_z.extend(Z_plot.tolist() + [np.nan])

            ax.plot3D(all_x, all_y, all_z, alpha=alpha, color=color, label=label, linewidth=1.4)

        plot_group(idx_surf, "tab:green", 0.3, "Absorbed by surface")
        plot_group(idx_toa, "tab:grey", 0.2, "Escaped (TOA)")
        plot_group(idx_atm, "tab:red", 0.3, "Absorbed by atmosphere")

        X_grid, Y_grid = np.meshgrid([-limit_x, limit_x], [-limit_y, limit_y])
        ax.plot_surface(
            X_grid, Y_grid, np.zeros_like(X_grid), color="gray", alpha=0.1, shade=False, zorder=-1
        )

        ax.view_init(elev=25, azim=-45)

        ax.set_title(title)
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Altitude Z [km]")

        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_y, limit_y)
        ax.set_zlim(0, toa_z)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")

        ax.legend(loc="upper right")

        fig.tight_layout()

        return fig

    def plot_2d_map(
        self,
        flux_map: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        title: str,
        label: str = "Normalized Flux",
    ):
        fig, ax = plt.subplots(figsize=(10, 10))

        mesh = ax.pcolormesh(x_centers, y_centers, flux_map.T, cmap=cmo.cm.solar, shading="nearest")
        ax.set_aspect("equal")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1)
        fig.colorbar(mesh, cax=cax, label=label, orientation="horizontal")

        ax.set_xlabel("Position X [km]")
        ax.set_ylabel("Position Y [km]")
        ax.set_title(title)
        sns.despine(ax=ax)

        fig.tight_layout()
        return fig

    def plot_surface_absorption_map(self, title: str = "Surface absorption map"):
        if "surface_absorption_map" not in self.ds:
            return None

        return self.plot_2d_map(
            self.ds["surface_absorption_map"].values,
            self.ds["x_surface"].values,
            self.ds["y_surface"].values,
            title,
        )

    def plot_flux_profile(self, title="Vertical radiative flux profile"):
        if "flux_down_profile" not in self.ds:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))
        z = self.ds["z_flux"].values

        flux_down = self.ds["flux_down_profile"].values
        flux_up = self.ds["flux_up_profile"].values
        net_flux = flux_down - flux_up

        ax.plot(flux_down, z, label=r"Downward flux ($F^\downarrow$)", color="#f7c522", linewidth=2)
        ax.plot(flux_up, z, label=r"Upward flux ($F^\uparrow$)", color="#1f78b4", linewidth=2)
        ax.plot(
            net_flux, z, label=r"Net flux ($F_{net}$)", color="black", linestyle="--", linewidth=2.5
        )

        ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Altitude Z [km]")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.legend()
        sns.despine(ax=ax)

        fig.tight_layout()
        return fig

    def plot_absorption_profile(self, title="Atmospheric absorption profile"):
        if "absorption_profile" not in self.ds:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))

        z_centers = self.ds.get("z_absorption")
        if z_centers is None:
            return None

        z_centers = z_centers.values
        profile = self.ds["absorption_profile"].values

        spacing = self.ds.attrs.get("vertical_resolution_km")
        if spacing is None:
            if len(z_centers) > 1:
                spacing = float(z_centers[1] - z_centers[0])
            else:
                spacing = 1.0

        ax.barh(
            z_centers,
            profile,
            height=spacing,
            align="center",
            color="#C25110",
            alpha=0.85,
            edgecolor="black",
        )

        ax.set_title(title)
        ax.set_xlabel("Normalized Absorption")
        ax.set_ylabel("Altitude Z [km]")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)

        fig.tight_layout()
        return fig

    def generate_all_figures(self):
        if "surface_absorption_map" in self.ds:
            fig = self.plot_surface_absorption_map()
            if fig:
                yield (fig, "surface_absorption_map")
                plt.close(fig)

        if "flux_down_profile" in self.ds:
            fig = self.plot_flux_profile()
            if fig:
                yield (fig, "vertical_flux_profile")
                plt.close(fig)

        if "paths" in self.ds:
            fig = self.plot_paths()
            if fig:
                yield (fig, "photon_paths_3d")
                plt.close(fig)

        if "downward_flux" in self.ds and "upward_flux" in self.ds:
            measure_z = self.ds["z_flux_map"].values
            x_centers = self.ds["x_flux"].values
            y_centers = self.ds["y_flux"].values

            incident_down = self.ds["downward_flux"].values
            incident_up = self.ds["upward_flux"].values

            for i, z_val in enumerate(measure_z):
                title_down = f"Downward radiative flux map\nHeight: {z_val} km"
                fig_down = self.plot_2d_map(
                    incident_down[i], x_centers, y_centers, title=title_down
                )
                if fig_down:
                    yield (fig_down, f"flux_map_downward_z_{z_val:g}km")
                    plt.close(fig_down)

                title_up = f"Upward radiative flux map\nHeight: {z_val} km"
                fig_up = self.plot_2d_map(incident_up[i], x_centers, y_centers, title=title_up)
                if fig_up:
                    yield (fig_up, f"flux_map_upward_z_{z_val:g}km")
                    plt.close(fig_up)

        if "absorption_profile" in self.ds:
            fig = self.plot_absorption_profile()
            if fig:
                yield (fig, "absorption_profile")
                plt.close(fig)
