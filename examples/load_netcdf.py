import xarray as xr

# Open the NetCDF file directly
ds = xr.open_dataset("results/demo001/atmorad_demo001_baseline.nc", engine="h5netcdf")

# Access variables and attributes ({detector_name}_{attribute_name})
map_2d = ds["surface_absorption_surface_absorption_map_2d"].values
total_reflected_energy = ds.attrs["fate_energy_outgoing_toa"]
