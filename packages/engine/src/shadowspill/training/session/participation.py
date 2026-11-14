"""Count declared validity on canonical CPU inputs, before padded worker routing."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from shadowspill_client import Datum
from shadowspill_client.protocol.definitions import Participation


def count_participation(batch: Sequence[Datum], declaration: Participation) -> int:
    """Count original logical positions once, never cooperating physical copies.

    Call after complete input validation on owned, canonical arrays. With no
    validity field each real Datum contributes, including zero-valued weights
    or losses. With a field, only its boolean validity matters. Padding is not
    present here and cannot turn an empty logical batch into a contribution.
    """
    if not declaration.path:
        return len(batch)
    count = 0
    for datum in batch:
        value: Any = datum.loss_inputs
        for part in declaration.path:
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"missing participation field: {declaration.path}")
            value = value[part]
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
            raise TypeError("canonical participation fields must be boolean CPU arrays")
        count += int(np.count_nonzero(value))
    return count
