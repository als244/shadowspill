"""The supplied text preset prepares tasks on token ids that are not all one id."""

import numpy as np

from shadowspill.training.components import UniformIntegers
from shadowspill.training.preparation.capture.representative import (
    representative_inputs,
)
from workload_tests.components.test_expert_components import prepared


def test_text_tasks_are_prepared_on_token_ids_drawn_over_the_vocabulary():
    _, inventory = prepared("qwen3moe")
    seen = []
    for template in inventory.templates.values():
        binding, pattern = template.binding.inputs, template.representative
        (sample,) = pattern.samples
        # The vocabulary of these settings has 97 ids.
        assert sample.values["inputs", "tokens"] == UniformIntegers(0, 96)
        with representative_inputs(binding, pattern, max_bytes=2 << 20) as inputs:
            tokens = np.array(inputs.fields["inputs", "tokens"]).reshape(-1)
        assert tokens.min() >= 0 and tokens.max() <= 96
        # One repeated id is the worst case for work that accumulates by token.
        assert len(np.unique(tokens)) > tokens.size // 2
        seen.append(tokens)
    assert seen
