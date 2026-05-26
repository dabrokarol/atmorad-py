"""
Registry for physics models and surface maps.

This module provides dictionaries and decorators to register classes so they
can be dynamically loaded from string names in the configuration files.
"""

import inspect
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

    def wrapper(target):
        if inspect.isclass(target):
            REFLECTION_MODELS[name] = target
            return target

        sig = inspect.signature(target)

        expected_params = list(sig.parameters.keys())[
            3:
        ]  # Skipping first three (direction, rand_1, rand_2)

        class DynamicReflection:
            def __init__(self, **kwargs):
                self.kwargs = {k: v for k, v in kwargs.items() if k in expected_params}

            def reflect(self, direction, rand_1, rand_2):
                return target(direction, rand_1, rand_2, **self.kwargs)

            def __call__(self, direction, rand_1, rand_2):
                return self.reflect(direction, rand_1, rand_2)

        REFLECTION_MODELS[name] = DynamicReflection
        return target

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

    def wrapper(target):
        if inspect.isclass(target):
            SCATTERING_MODELS[name] = target
            return target

        sig = inspect.signature(target)

        expected_params = list(sig.parameters.keys())[2:]

        class DynamicScattering:
            def __init__(self, **kwargs):
                self.kwargs = {k: v for k, v in kwargs.items() if k in expected_params}

            def scatter(self, rand_1, rand_2):
                return target(rand_1, rand_2, **self.kwargs)

            def __call__(self, rand_1, rand_2):
                return self.scatter(rand_1, rand_2)

        SCATTERING_MODELS[name] = DynamicScattering

        return target

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

    def wrapper(target):

        if inspect.isclass(target):
            SURFACE_MAPS[name] = {"class": target, "material_keys": material_keys}
            return target

        sig = inspect.signature(target)
        expected_params = list(sig.parameters.keys())[1:]  # Skipping 'pos'

        class DynamicSurfaceMap:
            def __init__(self, **kwargs):
                self.kwargs = {k: v for k, v in kwargs.items() if k in expected_params}

            def get_material_ids(self, pos):
                # Call user function
                return target(pos, **self.kwargs)

        SURFACE_MAPS[name] = {"class": DynamicSurfaceMap, "material_keys": material_keys}

        return target

    return wrapper


def register_detector(name: str, result_class: Any):
    """Decorator to register a detector model."""

    def wrapper(cls):
        DETECTORS[name] = cls

        # sets registry name inside the class for easier lookup
        result_class._registry_id = name
        DETECTOR_RESULTS[name] = result_class
        return cls

    return wrapper
