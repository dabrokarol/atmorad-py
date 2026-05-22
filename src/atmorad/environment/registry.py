SURFACE_MAPS = {}


def register_surface_map(name: str, material_keys: list[str]):
    """
    Decorator to register a surface map class.

    Args:
        name: The name used in the TOML file (e.g., "checkerboard").
        material_keys: The TOML keys expected for materials (e.g., ["material_a", "material_b"]).
    """

    def decorator(cls):
        SURFACE_MAPS[name] = {"class": cls, "material_keys": material_keys}
        return cls

    return decorator
