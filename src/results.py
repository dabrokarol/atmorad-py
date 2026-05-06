import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import seaborn as sns
import colorcet
import pandas as pd

from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap
from dataclasses import dataclass

sns.set_theme(style="ticks")

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

    def __post_init__(self):
        self.atmosphere_mask = (~self.space_mask) & (~self.surface_mask)

    def __str__(self):
        return f"photons left atmosphere: {np.count_nonzero(self.space_mask)}\n\
photons absorbed by surface: {np.count_nonzero(self.surface_mask)}\n\
photons absorbed by atmosphere {np.count_nonzero(self.atmosphere_mask)}\n"

    def plot_paths(self):    
        fig = plt.figure(figsize=(10,10))
        sns.set_style('ticks')
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

        ax.set_title('Paths of sampled photons')
        ax.set_xlabel('Pos x')
        ax.set_ylabel('Pos y')
        ax.set_zlabel('Pos z')

        ax.invert_zaxis()
        ax.legend()
        return fig

    def surface_plot(self):
        pos = self.final_positions[:, self.surface_mask]
        pos = pos[:, (pos[0]>-30) & (pos[0]<30) & (pos[1] > -30) & (pos[1]<30)]
        joint_plot = sns.jointplot(x=pos[0], y=pos[1], kind="hex", cmap='viridis')
        return joint_plot.figure

    