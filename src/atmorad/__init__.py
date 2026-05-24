from .api import load, run
from .config import SimConfig
from .detectors import BaseDetector
from .environment import BaseSurfaceMap, Scene
from .models import PhotonBatch
from .models.results import (
    BaseResult,
)
from .models.results import (
    attr_field as nc_attr,
)
from .models.results import (
    coord_field as nc_coord,
)
from .models.results import (
    data_field as nc_data,
)
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
    # Results handling
    "BaseResult",
    "nc_attr",
    "nc_data",
    "nc_coord",
]
