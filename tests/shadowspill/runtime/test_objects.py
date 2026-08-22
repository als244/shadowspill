from __future__ import annotations

from shadowspill.runtime import ObjectRef


class _Owner:
    def __init__(self) -> None:
        self.released: list[ObjectRef] = []

    def _release_object_reference(self, reference: ObjectRef) -> None:
        self.released.append(reference)


def test_object_reference_is_pool_neutral_and_closes_once() -> None:
    owner = _Owner()
    reference = ObjectRef(owner, object_id=17, size_bytes=4096, handle=91)

    assert reference.object_id == 17
    assert reference.size_bytes == 4096
    assert reference._belongs_to(owner)
    assert reference._require_handle() == 91

    reference.close()
    reference.close()

    assert reference.closed
    assert owner.released == [reference]


def test_object_reference_context_manager_releases_ownership() -> None:
    owner = _Owner()
    with ObjectRef(owner, object_id=1, size_bytes=0, handle=3) as reference:
        assert not reference.closed

    assert reference.closed
    assert owner.released == [reference]
