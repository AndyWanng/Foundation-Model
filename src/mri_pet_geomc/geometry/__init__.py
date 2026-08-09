"""FEM geometry, Parseval resolvent frames, and coordinate interpolation."""

from .fem import FEMGeometry, build_fem_geometry
from .frame import FrameAnalysis, FrameDiagnostics, ParsevalResolventFrame
from .interpolation import (
    FixedSpatialInterpolator,
    InterpolationPlan,
    build_interpolator,
    token_centers_from_reference,
)

__all__ = [
    "FEMGeometry",
    "FixedSpatialInterpolator",
    "FrameAnalysis",
    "FrameDiagnostics",
    "InterpolationPlan",
    "ParsevalResolventFrame",
    "build_fem_geometry",
    "build_interpolator",
    "token_centers_from_reference",
]