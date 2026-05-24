import json
import logging

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_theme(style="ticks", rc={"font.family": "serif"})


class ResultAnalyzer:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.det_types = json.loads(str(ds.attrs.get("_detector_types", "{}")))

    def experiment_summary(self) -> str:
        experiment_name = self.ds.attrs.get("experiment_name", "")
        total_time = self.ds.attrs.get("engine_simulation_time_s", 0.0)
        cpu_time = self.ds.attrs.get("engine_cpu_time_s", 0.0)
        total_photons = int(self.ds.attrs.get("num_photons", 0))

        fate_prefix = next((p for p, c in self.det_types.items() if c == "FateResult"), "fate")

        escaped_toa = self.ds.attrs.get(f"{fate_prefix}_energy_escaped_toa", 0.0)
        abs_surf = self.ds.attrs.get(f"{fate_prefix}_energy_absorbed_surface", 0.0)
        abs_atm = self.ds.attrs.get(f"{fate_prefix}_energy_absorbed_atmosphere", 0.0)

        escaped_toa_pct = escaped_toa * 100.0
        absorbed_surf_pct = abs_surf * 100.0
        absorbed_atm_pct = abs_atm * 100.0

        balance = escaped_toa_pct + absorbed_surf_pct + absorbed_atm_pct

        return "\n".join(
            [
                f"\n---- Simulation Summary: {experiment_name} ----",
                f"Time: {total_time:.2f}s (Total) | {cpu_time:.2f}s (CPU)",
                f"Total Photons Simulated: {total_photons:_}\n",  # Używamy self.total_photons (int)
                "Energy Distribution:",
                f"  {'Escaped (TOA)':<21}: {escaped_toa_pct:>6.2f}%",
                f"  {'Surface Absorbed':<21}: {absorbed_surf_pct:>6.2f}%",
                f"  {'Atmosphere Absorbed':<21}: {absorbed_atm_pct:>6.2f}%",
                "  " + "-" * 30,
                f"  {'Energy Balance':<21}: {balance:>6.2f}%\n",
            ]
        )

    def _infer_domain_size(self, paths_array: np.ndarray) -> tuple[float, float]:
        """Dynamically infers the physical domain limits (Lx, Ly) from the dataset."""
        Lx = self.ds.attrs.get("domain_size_x_km")
        Ly = self.ds.attrs.get("domain_size_y_km")

        if Lx is None or Ly is None:
            x_coords = [k for k in self.ds.coords if isinstance(k, str) and k.endswith("_x")]
            y_coords = [k for k in self.ds.coords if isinstance(k, str) and k.endswith("_y")]

            if x_coords and Lx is None:
                x_vals = self.ds.coords[x_coords[0]].values
                if len(x_vals) > 1:
                    Lx = float(x_vals.max() - x_vals.min() + (x_vals[1] - x_vals[0]))

            if y_coords and Ly is None:
                y_vals = self.ds.coords[y_coords[0]].values
                if len(y_vals) > 1:
                    Ly = float(y_vals.max() - y_vals.min() + (y_vals[1] - y_vals[0]))

        if Lx is None:
            Lx = float(np.nanmax(np.abs(paths_array[:, :, 0]))) * 2.0
        if Ly is None:
            Ly = float(np.nanmax(np.abs(paths_array[:, :, 1]))) * 2.0

        return max(Lx, 1.0), max(Ly, 1.0)

    def plot_paths(self, prefix: str, title: str = "Sample 3D photon paths", max_paths: int = 500):
        path_var = f"{prefix}_sample_paths_3d"
        if path_var not in self.ds:
            return None

        paths = self.ds[path_var].values
        num_paths = paths.shape[0]

        if num_paths == 0:
            return None

        if num_paths > max_paths:
            logging.warning(
                f"Limiting 3D plot to {max_paths} paths (out of {num_paths}) to prevent freezing."
            )
            paths = paths[:max_paths]
            num_paths = max_paths

        absorbed_surface = self.ds[f"{prefix}_sample_absorbed_surface"].values[:num_paths]
        escaped_toa = self.ds[f"{prefix}_sample_escaped_toa"].values[:num_paths]
        toa_z = self.ds.attrs.get(f"{prefix}_toa_z", 10.0)

        Lx, Ly = self._infer_domain_size(paths)
        limit_x, limit_y = Lx / 2.0, Ly / 2.0

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        labeled_surface, labeled_above_toa, labeled_atmosphere = False, False, False

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

            if absorbed_surface[i]:
                color, alpha = "tab:green", 0.3
                lbl = "Absorbed by surface" if not labeled_surface else None
                labeled_surface = True
            elif escaped_toa[i]:
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
        ax.set_zlim(0, toa_z)

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
        fig, ax = plt.subplots(figsize=(8, 7))
        X, Y = np.meshgrid(x_centers, y_centers)

        mesh = ax.pcolormesh(X, Y, flux_map.T, cmap=cmo.cm.solar, shading="nearest")  # type: ignore
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, label=label, orientation="horizontal", pad=0.1)
        ax.set_xlabel("Position X [km]")
        ax.set_ylabel("Position Y [km]")
        ax.set_title(title, fontsize=16)

        return fig

    def plot_surface_absorption_map(self, prefix: str, title: str = "Surface Absorption Map"):
        var_name = f"{prefix}_surface_absorption_map_2d"
        if var_name not in self.ds:
            logging.warning(f"Warning: No surface absorption map found for prefix '{prefix}'.")
            return None

        return self.plot_2d_map(
            self.ds[var_name].values,
            self.ds[f"{prefix}_x"].values,
            self.ds[f"{prefix}_y"].values,
            title,
        )

    def plot_flux_profile(self, prefix: str, title="Vertical Flux Profile"):
        var_down = f"{prefix}_flux_down"
        if var_down not in self.ds:
            return None

        fig, ax = plt.subplots(figsize=(8, 10))
        z = self.ds[f"{prefix}_z"].values

        flux_down = self.ds[var_down].values
        flux_up = self.ds[f"{prefix}_flux_up"].values
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
