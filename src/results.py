import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import seaborn as sns
import colorcet
import pandas as pd

from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap

sns.set_theme(style="ticks")

class Results:
    def __init__(self, last_positions, space_mask, surface_mask, layer_idx, sample_paths) -> None:
        self.last_positions = last_positions
        self.space_mask = space_mask
        self.surface_mask = surface_mask
        self.atmosphere_mask = (~space_mask) & (~surface_mask)
        self.sample_paths = sample_paths
        self.layer_idx = layer_idx

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
        pos = self.last_positions[:, self.surface_mask]
        pos = pos[:, (pos[0]>-30) & (pos[0]<30) & (pos[1] > -30) & (pos[1]<30)]
        joint_plot = sns.jointplot(x=pos[0], y=pos[1], kind="hex", cmap='viridis')
        return joint_plot.figure

    