"""
Registry for physics models and surface maps.

This module provides dictionaries and decorators to register classes so they
can be dynamically loaded from string names in the configuration files.
"""

from typing import Any

SCATTERING_MODELS: dict[str, Any] = {}
REFLECTION_MODELS: dict[str, Any] = {}
SURFACE_MAPS: dict[str, Any] = {}
DETECTORS: dict[str, Any] = {}
DETECTOR_RESULTS: dict[str, Any] = {}


def register_reflection(name: str):
    """
    Decorator to register a surface reflection model.

    Args:
        name (str): The string name used to identify this reflection model
            in the configuration file (e.g., "lambertian", "specular").

    Returns:
        Callable: The decorator function that adds the class to REFLECTION_MODELS.
    """

    def wrapper(cls):
        REFLECTION_MODELS[name] = cls
        return cls

    return wrapper


def register_scattering(name: str):
    """
    Decorator to register an atmospheric scattering phase function.

    Args:
        name (str): The string name used to identify this scattering model
            in the configuration file (e.g., "rayleigh", "hg", "isotropic").

    Returns:
        Callable: The decorator function that adds the class to SCATTERING_MODELS.
    """

    def wrapper(cls):
        SCATTERING_MODELS[name] = cls
        return cls

    return wrapper


def register_surface_map(name: str, material_keys: list[str]):
    """
    Decorator to register a procedural surface map layout.

    Args:
        name (str): The string name used to identify this map in the TOML
            configuration file (e.g., "checkerboard", "split-half-x").
        material_keys (list[str]): The specific TOML keys expected to define
            the materials for this map layout (e.g., ["material_a", "material_b"]).

    Returns:
        Callable: The decorator function that adds the map data to SURFACE_MAPS.
    """

    def decorator(cls):
        SURFACE_MAPS[name] = {"class": cls, "material_keys": material_keys}
        return cls

    return decorator


def register_detector(name: str, result_class: Any):
    """Decorator to register a detector model."""

    def wrapper(cls):
        DETECTORS[name] = cls

        # sets registry name inside the class for easier lookup
        result_class._registry_id = name
        DETECTOR_RESULTS[name] = result_class
        return cls

    return wrapper
