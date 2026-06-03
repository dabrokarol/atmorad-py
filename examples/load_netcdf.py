import xarray as xr

# open NetCDF file
ds = xr.open_dataset("results/demo001/atmorad_demo001_baseline.nc", engine="h5netcdf")

# access arrays
map_2d = ds["surface_absorption_map"].values
flux_profile = ds["flux_down_profile"].values

# access numbers
total_reflected_energy = ds["energy_toa_outgoing"].item()
total_absorbed_surf = ds["energy_surface_absorbed"].item()

# access attributes
num_photons = ds.attrs["num_photons"]
sim_time = ds.attrs["simulation_time_s"]
active_detectors = ds.attrs["active_detectors"]
