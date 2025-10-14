"""Top-level package for flypca."""

from .io import load_trials  # noqa: F401
from .lagpca import LagPCAResult, fit_lag_pca_for_trials, project_trial  # noqa: F401
from .features import compute_feature_table  # noqa: F401

__all__ = [
    "load_trials",
    "LagPCAResult",
    "fit_lag_pca_for_trials",
    "project_trial",
    "compute_feature_table",
]
