"""Physical memory is not the same capacity as a nested allocation cgroup."""

from pathlib import Path

import pytest

from shadowspill.orchestration.launch.host_memory import host_memory


def fixture(root, *, kind="cgroup2", membership="/job/step", mounted_root="/"):
    proc, mount = root / "proc", root / "cgroup with spaces"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    (proc / "meminfo").write_text("MemTotal: 200 kB\nMemAvailable: 100 kB\n")
    (proc / "self/cgroup").write_text(
        f"0::{membership}\n" if kind == "cgroup2" else f"7:cpu,memory:{membership}\n"
    )
    path = str(mount).replace(" ", r"\040")
    (proc / "self/mountinfo").write_text(
        f"31 22 0:27 {mounted_root} {path} rw - {kind} cgroup rw,memory\n"
    )
    current = mount / Path(membership).relative_to(mounted_root)
    current.mkdir(parents=True, exist_ok=True)
    return proc, mount, current


def limit(path, capacity, used, *, kind="cgroup2"):
    a, b = (
        ("memory.max", "memory.current")
        if kind == "cgroup2"
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    (path / a).write_text(str(capacity))
    (path / b).write_text(str(used))


def test_every_visible_parent_limit_and_usage_constrains_availability(tmp_path):
    proc, mount, current = fixture(tmp_path)
    limit(current, 80 << 10, 30 << 10)
    limit(current.parent, 120 << 10, 100 << 10)
    (mount / "memory.current").write_text("180000")  # The v2 root has no memory.max.
    memory = host_memory(proc_root=proc)
    assert memory.physical_bytes == 200 << 10
    assert memory.physical_available_bytes == 100 << 10
    assert memory.capacity_bytes == 80 << 10
    assert memory.available_bytes == 20 << 10
    assert [item.scope for item in memory.limits] == [str(current), str(current.parent)]
    limit(current.parent, 50 << 10, 60 << 10)
    assert host_memory(proc_root=proc).available_bytes == 0


def test_unlimited_or_disabled_memory_controller_retains_physical_snapshot(tmp_path):
    proc, _, current = fixture(tmp_path)
    limit(current, "max", 30 << 10)
    memory = host_memory(proc_root=proc)
    assert memory.capacity_bytes == 200 << 10
    assert memory.available_bytes == 100 << 10 and not memory.limits
    (proc / "self/cgroup").write_text("7:cpu:/job/step\n")
    (proc / "self/mountinfo").unlink()
    assert host_memory(proc_root=proc) == memory


def test_v1_bind_mount_maps_membership_and_retains_ancestor_limits(tmp_path):
    proc, mount, current = fixture(
        tmp_path, kind="cgroup", membership="/job/step", mounted_root="/job"
    )
    limit(current, 64 << 10, 20 << 10, kind="cgroup")
    limit(mount, 128 << 10, 100 << 10, kind="cgroup")
    memory = host_memory(proc_root=proc)
    assert memory.capacity_bytes == 64 << 10
    assert memory.available_bytes == 28 << 10


def test_wider_mount_prevents_a_bind_mount_hiding_parent_limits(tmp_path):
    proc, _, current = fixture(tmp_path)
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    (narrow / "step").mkdir()
    limit(narrow / "step", 80 << 10, 30 << 10)
    limit(current, 80 << 10, 30 << 10)
    limit(current.parent, 64 << 10, 60 << 10)
    path = proc / "self/mountinfo"
    path.write_text(
        f"32 22 0:27 /job {narrow} rw - cgroup2 cgroup rw\n" + path.read_text()
    )
    assert host_memory(proc_root=proc).available_bytes == 4 << 10


def test_v2_bind_mount_root_is_not_the_unbounded_hierarchy_root(tmp_path):
    proc, mount, _ = fixture(tmp_path, mounted_root="/job")
    (mount / "memory.current").write_text("100")
    with pytest.raises(ValueError, match="incomplete"):
        host_memory(proc_root=proc)


@pytest.mark.parametrize(
    "failure", ["mount", "directory", "partial", "counter", "escape"]
)
def test_inaccessible_or_malformed_cgroup_never_becomes_whole_host_capacity(
    tmp_path, failure
):
    proc, _, current = fixture(tmp_path)
    limit(current, 80 << 10, 30 << 10)
    if failure == "mount":
        (proc / "self/mountinfo").write_text("")
    elif failure == "directory":
        (proc / "self/cgroup").write_text("0::/missing\n")
    elif failure == "partial":
        (current / "memory.current").unlink()
    elif failure == "counter":
        (current / "memory.current").write_text("-1")
    else:
        (proc / "self/cgroup").write_text("0::/../../other\n")
    with pytest.raises(ValueError):
        host_memory(proc_root=proc)


@pytest.mark.parametrize(
    "contents",
    [
        "MemTotal: 100 kB\nMemAvailable: 200 kB\n",
        "MemTotal: 100 kB\n",
        "MemTotal: 100 kB\nMemAvailable: 20 MB\n",
        "MemTotal: 100 kB\nMemAvailable: 20 kB\nMemAvailable: 10 kB\n",
        "x" * ((16 << 10) + 1),
    ],
)
def test_physical_memory_fields_units_duplicates_and_read_bounds(tmp_path, contents):
    proc, _, _ = fixture(tmp_path)
    (proc / "meminfo").write_text(contents)
    with pytest.raises(ValueError):
        host_memory(proc_root=proc)
