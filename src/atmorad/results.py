import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean as cmo

from dataclasses import dataclass

sns.set_theme(style="ticks", rc={"font.family": "serif"})

@dataclass
class Results:
    final_positions: np.ndarray
    space_mask: np.ndarray
    surface_mask: np.ndarray
    layer_idx: np.ndarray
    sample_paths: dict
    surface_hits: np.ndarray
    scatter_counts: np.ndarray
    measure_z: np.ndarray
    flux_up: np.ndarray
    flux_down: np.ndarray
    sim_duration_s: float

    def __post_init__(self):
        self.atmosphere_mask = (~self.space_mask) & (~self.surface_mask)
    
    @classmethod
    def merge_all(cls, results_list: list['Results']) -> 'Results':
        if len(results_list) == 1:
            return results_list[0]

        final_positions = np.concatenate([r.final_positions for r in results_list], axis=1)
        space_mask = np.concatenate([r.space_mask for r in results_list])
        surface_mask = np.concatenate([r.surface_mask for r in results_list])
        layer_idx = np.concatenate([r.layer_idx for r in results_list])
        scatter_counts = np.concatenate([r.scatter_counts for r in results_list])

        sample_paths = {}
        idx = 0
        for r in results_list:
            for path in r.sample_paths.values():
                sample_paths[idx] = path
                idx += 1

        valid_hits = [r.surface_hits for r in results_list if r.surface_hits.size > 0]
        if not valid_hits:
            surface_hits = np.array([])
        else:
            surface_hits = np.concatenate(valid_hits, axis=1)

        flux_up = np.sum([r.flux_up for r in results_list], axis=0)
        flux_down = np.sum([r.flux_down for r in results_list], axis=0)
        
        sim_duration_s = sum(r.sim_duration_s for r in results_list)

        return cls(
            final_positions=final_positions,
            space_mask=space_mask,
            surface_mask=surface_mask,
            layer_idx=layer_idx,
            sample_paths=sample_paths,
            surface_hits=surface_hits,
            scatter_counts=scatter_counts,
            measure_z=results_list[0].measure_z,
            flux_up=flux_up,
            flux_down=flux_down,
            sim_duration_s=sim_duration_s
        )

    def summary(self):
        return (
            f"---- Simulation results ----\n"
            f"photons left atmosphere: {np.count_nonzero(self.space_mask)}\n"
            f"photons absorbed by surface: {np.count_nonzero(self.surface_mask)}\n"
            f"photons absorbed by atmosphere {np.count_nonzero(self.atmosphere_mask)}\n"
        )

    def plot_paths(self, title: str = "Sample 3D photon paths", limit_xy=100):    
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')

        labeled_surface, labeled_space, labeled_atmosphere = False, False, False

        for i, h in self.sample_paths.items(): 
            if self.surface_mask[i]:
                color = 'tab:green'
                alpha=0.3
                lbl = 'Absorbed by surface' if not labeled_surface else None
                labeled_surface = True
            elif self.atmosphere_mask[i]:
                color = 'tab:red'
                alpha=0.3
                lbl = 'Absorbed by atmosphere' if not labeled_atmosphere else None
                labeled_atmosphere = True
            else:
                color = 'tab:grey'
                alpha = 0.2
                lbl = 'Escaped atmosphere' if not labeled_space else None
                labeled_space = True
            X, Y, Z = np.array(h).T
            ax.plot3D(X, Y, Z, alpha=alpha, color=color, label=lbl)

        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Pos x')
        ax.set_ylabel('Pos y')
        ax.set_zlabel('Pos z')

        ax.set_xlim(-limit_xy, limit_xy)
        ax.set_ylim(-limit_xy, limit_xy)

        ax.invert_zaxis()
        ax.legend()
        return fig

    def surface_absorption_plot(self, title: str = 'Photons absorbed by the ground', limit=40):
        return self._2dhexplot(self.final_positions[:, self.surface_mask], title, limit)
        
    def surface_flux_plot(self, title: str ='Downward flux near the ground', limit: float = 40):
        return self._2dhexplot(self.surface_hits, title, limit)
    
    def _2dhexplot(self, pos, title, limit):
        if not pos.any():
            return plt.figure()
        plot_mask = (pos[0] > -limit) & (pos[0] < limit) & (pos[1] > -limit) & (pos[1] < limit)
        pos = pos[:, plot_mask]
        color = cmo.cm.solar(2) # type: ignore
        joint_plot = sns.jointplot(x=pos[0], y=pos[1], kind="hex", cmap=cmo.cm.solar, color=color) # type: ignore

        joint_plot.ax_joint.set_xbound(-limit, limit)
        joint_plot.ax_joint.set_ybound(-limit, limit)
        mappable = joint_plot.ax_joint.collections[0]
        joint_plot.ax_joint.figure.colorbar(mappable, label='Photon count', ax=[joint_plot.ax_joint, joint_plot.ax_marg_x, joint_plot.ax_marg_y], orientation='horizontal')
        joint_plot.ax_joint.set_xlabel('Position (X)')
        joint_plot.ax_joint.set_ylabel('Position (Y)')
        joint_plot.ax_joint.set_title(title, fontsize=14, pad=70)

        return joint_plot.figure
    
    def plot_flux_profile(self, title='Vertical flux profile'):
        fig, ax = plt.subplots(figsize=(8, 10))
        net_flux = self.flux_down - self.flux_up

        ax.plot(self.flux_down, self.measure_z, label=r'Downward flux ($F^\downarrow$)', color='tab:blue', linewidth=2)
        ax.plot(self.flux_up, self.measure_z, label=r'Upward flux ($F^\uparrow$)', color='tab:orange', linewidth=2)
        ax.plot(net_flux, self.measure_z, label=r'Net flux ($F_{net}$)', color='black', linestyle='--', linewidth=2.5)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Flux [photon count]", fontsize=12)
        ax.set_ylabel("Height (Z)", fontsize=12)

        ax.invert_yaxis()
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)
        
        ax.fill_betweenx(self.measure_z, 0, net_flux, color='gray', alpha=0.1)

        fig.tight_layout()
        return fig

    def plot_scattering_histogram(self, title='Scattering Counts'):
        fig, ax = plt.subplots(figsize=(10, 6))
        color = cmo.cm.solar(2) # type: ignore
        sns.histplot(self.scatter_counts, ax=ax, color=color, discrete=True)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Number of scatterings')
        ax.set_ylabel('Number of photons')
        return fig    
    