"""The reference check is shared by every harness; FSDP2 uses it with the
default access to plain tensors and shards."""

from baselines.common.reference import Access, Reference, canonical_name

__all__ = ["Access", "Reference", "canonical_name"]
