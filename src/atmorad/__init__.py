from .api import load, run
from .config import SimConfig
from .constants import X, Y, Z
from .detectors import BaseDetector
from .environment import BaseSurfaceMap, Scene
from .models import PhotonBatch
from .physics import Scattering, SurfaceReflection
from .registry import (
    register_detector,
    register_reflection,
    register_scattering,
    register_surface_map,
)

__all__ = [
    # Main API
    "run",
    "load",
    # Registry decorators
    "register_detector",
    "register_reflection",
    "register_scattering",
    "register_surface_map",
    # Base Classes & Core Types for Plugins
    "BaseDetector",
    "Scattering",
    "SurfaceReflection",
    "BaseSurfaceMap",
    "SimConfig",
    "Scene",
    "PhotonBatch",
    # Constants
    "X",
    "Y",
    "Z",
]
