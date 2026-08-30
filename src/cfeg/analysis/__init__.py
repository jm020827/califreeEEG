"""Analysis utilities for subject-level and OOD-cell research results."""

from cfeg.analysis.ood_coverage import compare_ood_coverage, paired_subject_inference
from cfeg.analysis.primary_aggregate import aggregate_primary_confirmatory

__all__ = [
    "aggregate_primary_confirmatory",
    "compare_ood_coverage",
    "paired_subject_inference",
]
