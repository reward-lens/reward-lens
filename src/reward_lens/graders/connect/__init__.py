"""The connector (D-65): four callable shapes recognised by signature, and two shapes refused.

Every adapter, audit and trace in the product enters through what this module recognises, and the
whole point of D-65 is that a shape is recognised by running a fixture rather than by importing the
framework that defines it. Nothing under here imports `trl`, `verifiers`, `transformers` or `verl`.

Owned by P-CONNECT.
"""

from .bind import ConnectedGrader, bind
from .detect import (
    ProjectDetection,
    SILENT_MODES,
    detect,
    detect_project,
    probe,
    render_detect_transcript,
    signature_text,
)
from .errors import SHAPE_UNSUPPORTED_PAGE, ShapeUnsupported, bad_override
from .shapes import DECLARED_SHAPES, Detection, Finding, Shape, shape_for_declared
from .verifiers_discriminator import GROUP_INDICATORS, bound_names, is_group_rubric

__all__ = [
    "ConnectedGrader",
    "DECLARED_SHAPES",
    "Detection",
    "Finding",
    "GROUP_INDICATORS",
    "ProjectDetection",
    "SHAPE_UNSUPPORTED_PAGE",
    "SILENT_MODES",
    "Shape",
    "ShapeUnsupported",
    "bad_override",
    "bind",
    "bound_names",
    "detect",
    "detect_project",
    "is_group_rubric",
    "probe",
    "render_detect_transcript",
    "shape_for_declared",
    "signature_text",
]
