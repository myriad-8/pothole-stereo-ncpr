"""Model registry: one file per model."""
from src.models.baseline import build_baseline
from src.models.ncpr_v1 import build_ncpr_v1
from src.models.proposed import build_proposed

REGISTRY = {"baseline": build_baseline, "ncpr_v1": build_ncpr_v1, "proposed": build_proposed}
