def merge_incremental(first: dict, second: dict) -> dict:
    if not first:
        return second

    for key in first.keys():
        if key not in second:
            continue
        if key in [
            "measure_z",
            "layer_boundaries_z",
            "x_edges",
            "y_edges",
            "incident_flux_heights_km",
            "sample_paths",
        ]:
            continue
        elif key in [
            "flux_up",
            "flux_down",
            "surface_absorption_map_2d",
            "toa_flux_map_2d",
            "heating_profile_1d",
            "scatter_counts",
            "cpu_time_s",
            "photons_absorbed_surface",
            "photons_absorbed_atmosphere",
            "photons_escaped_toa",
        ]:
            first[key] += second[key]
        elif key in ["incident_flux_down_maps_2d", "incident_flux_up_maps_2d"]:
            for z_measure in first[key].keys():
                first[key][z_measure] += second[key][z_measure]

    for key in second.keys():
        if key not in first:
            first[key] = second[key]
    return first
