from dataclasses import dataclass

@dataclass
class SimConfig:
    num_photons: int = 100_000
    num_track: int = 100
    random_seed: int = 42
    theta_sun_deg: float = 60
    phi_sun_deg: float = 0
    flux_measure_spacing: float = 1
    cpu_cores: int = 4
    
    
class LambertianReflection:
    def __init__(self):
        pass
    
class SpecularReflection:
    def __init__(self, roughness: float):
        self.roughness = roughness

class SurfaceMaterial:
    def __init__(self, albedo: float, reflection_model):
        self.albedo = albedo
        self.reflection_model = reflection_model
        
REFLECTION_MODELS = {
    "lambertian": LambertianReflection,
    "specular": SpecularReflection
}

import tomllib

def parse_surface_materials(toml_dict):
    parsed_materials = {}

    for name, properties in toml_dict["surface_materials"].items():
        albedo = properties["albedo"]
        ref_data = properties["reflection"]

        ref_type = ref_data.pop("type")

        model_class = REFLECTION_MODELS[ref_type] 
        reflection_instance = model_class(**ref_data) 
        
        parsed_materials[name] = SurfaceMaterial(albedo, reflection_instance)
        
    return parsed_materials

# with open("config.toml", "rb") as f:
#     config_data = tomllib.load(f)

# materials = parse_surface_materials(config_data)

# print(materials["ocean"].reflection_model.roughness)
