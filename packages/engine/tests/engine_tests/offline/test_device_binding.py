from dataclasses import replace

import pytest

from shadowspill.artifacts.device import require_compiled_device

from ._example import program


def test_compiled_device_binding_survives_archive_and_rejects_wrong_ordinal(tmp_path):
    saved = replace(program(), bindings={"compiled_device_index": 0})
    saved.save(tmp_path / "program.ssp")
    restored = type(saved).load(tmp_path / "program.ssp")
    require_compiled_device(restored, 0)
    with pytest.raises(ValueError, match="cannot remap device indices"):
        require_compiled_device(restored, 1)
    with pytest.raises(ValueError, match="no recorded compiler device binding"):
        require_compiled_device(program(), 0)
