import matplotlib.pyplot as plt

import atmorad

# Load the completed simulation
results = atmorad.load("results/demo001")

# Access physical data as NumPy arrays
map_2d = results.detector_results["surface_absorption"].surface_absorption_map_2d

# Analyze or plot
plt.imshow(map_2d)
plt.title(f"Flux Map for {results.config.metadata.experiment_name}")
plt.show()
