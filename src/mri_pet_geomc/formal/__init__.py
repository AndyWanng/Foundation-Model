"""Formal workstation pretraining path.

The original :mod:`mri_pet_geomc` feasibility pipeline is intentionally kept
intact.  Modules below this package implement the independent, resumable
workstation workflow for the 24k-observation MRI/PET run.
"""

from .config import FormalConfigError, config_digest, load_formal_config

__all__ = ["FormalConfigError", "config_digest", "load_formal_config"]
