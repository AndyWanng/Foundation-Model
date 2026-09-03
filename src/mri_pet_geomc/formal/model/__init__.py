"""Formal 24k-scale model components.

These modules coexist with the preliminary model and do not change its legacy
contracts.  The data/training pipeline can opt into this package explicitly.
"""

from .adapters import (
    ConditionedFeatureAdapter3D,
    MRIPETResidualStems,
    NearIdentityResidualStem,
    SafeConditionedEncoderWrapper,
)
from .condition import (
    AcquisitionConditionEncoder,
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    ConditionBatch,
    MetadataVocabulary,
)
from .checkpointing import (
    ActivationCheckpointingReport,
    CheckpointedEncoderStage,
    activation_checkpointing_status,
    configure_sat3d_activation_checkpointing,
)
from .ema import EncodedObservation, FormalEncoderBundle, OnlineEMABundle
from .foundation import (
    FormalFoundationOutput,
    FormalGeoMCJEPAModel,
    FormalObservationBatch,
    PackedCompanionBatch,
)
from .fusion import AnchorCentricField, AnchorCentricFusion
from .predictor import PhysicalPositionEncoding3D, QueryAwareJEPA3DPredictor
from .projector import ContentGatedTokenProjector

__all__ = [
    "AcquisitionConditionEncoder",
    "ActivationCheckpointingReport",
    "AnchorCentricField",
    "AnchorCentricFusion",
    "CATEGORICAL_FIELDS",
    "CONTINUOUS_FIELDS",
    "ConditionBatch",
    "ConditionedFeatureAdapter3D",
    "CheckpointedEncoderStage",
    "ContentGatedTokenProjector",
    "EncodedObservation",
    "FormalEncoderBundle",
    "FormalFoundationOutput",
    "FormalGeoMCJEPAModel",
    "FormalObservationBatch",
    "MRIPETResidualStems",
    "MetadataVocabulary",
    "NearIdentityResidualStem",
    "OnlineEMABundle",
    "PackedCompanionBatch",
    "PhysicalPositionEncoding3D",
    "QueryAwareJEPA3DPredictor",
    "SafeConditionedEncoderWrapper",
    "activation_checkpointing_status",
    "configure_sat3d_activation_checkpointing",
]
