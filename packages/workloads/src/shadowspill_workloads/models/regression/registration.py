"""The small non-text model uses the same installed registration path."""

from shadowspill_workloads.models.registration import Family

FAMILY = Family(
    name="regression",
    training="shadowspill_workloads.models.regression.training:register",
)
