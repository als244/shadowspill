"""The NVIDIA environment's preferred versions from NVIDIA's two lock files:
which lock a package's version comes from, what is left to its source, and
what a resolution departed from."""

from baselines.scripts.nvidia_versions import departures, preferences

BRIDGE = """
version = 1

[[package]]
name = "shared-pkg"
version = "2.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pinned-by-automodel"
version = "5.20"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "engine"
version = "2.20"
source = { git = "https://example.org/engine.git?rev=abc" }

[[package]]
name = "bridge-only"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
"""

AUTOMODEL = """
version = 1

[[package]]
name = "nemo-automodel"
version = "0.8.0"
source = { editable = "." }

[package.metadata]
requires-dist = [{ name = "pinned-by-automodel", specifier = "==5.18" }]

[[package]]
name = "shared-pkg"
version = "1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pinned-by-automodel"
version = "5.18"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "engine"
version = "2.19"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "twice"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version < '3.0'"]

[[package]]
name = "twice"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version >= '3.0'"]
"""


def locks(tmp_path):
    bridge, automodel = tmp_path / "bridge.lock", tmp_path / "automodel.lock"
    bridge.write_text(BRIDGE)
    automodel.write_text(AUTOMODEL)
    return str(bridge), str(automodel)


def test_bridges_lock_comes_first_and_automodels_where_it_must(tmp_path):
    chosen, held, _, left_open = preferences(*locks(tmp_path), set(), set())
    assert chosen["shared-pkg"] == "2.0"
    assert chosen["bridge-only"] == "1.0"
    # AutoModel's own exact pin excludes Bridge's version.
    assert chosen["pinned-by-automodel"] == "5.18"
    # Locked twice, taken in the version whose markers hold here.
    assert chosen["twice"] == "2.0"
    # Bridge takes it from git: its source pins it, not AutoModel's index.
    assert "engine" not in chosen
    assert not held and not left_open


def test_a_package_the_resolution_moved_is_preferred_at_automodels(tmp_path):
    chosen, *_ = preferences(*locks(tmp_path), set(), {"shared-pkg"})
    assert chosen["shared-pkg"] == "1.0"


def test_a_kept_package_is_left_to_its_own_pin(tmp_path):
    chosen, *_ = preferences(*locks(tmp_path), {"bridge-only"}, set())
    assert "bridge-only" not in chosen


def test_departures_name_what_the_resolution_moved(tmp_path):
    preferred, resolved = tmp_path / "preferred.txt", tmp_path / "resolved.txt"
    preferred.write_text("shared-pkg==2.0\nother==1.0\n")
    resolved.write_text("-e file:///src\nshared_pkg==2.1\nother==1.0\nnew==3\n")
    assert departures(str(preferred), str(resolved)) == [("shared-pkg", "2.0", "2.1")]
