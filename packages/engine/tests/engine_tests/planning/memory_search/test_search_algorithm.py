"""A search registers by its name; an abstract intermediate needs none."""

from abc import abstractmethod

import pytest

from shadowspill.planning.memory_search.algorithm import SearchAlgorithm


def test_abstract_subclass_needs_no_name_and_is_not_registered():
    before = dict(SearchAlgorithm._NAMED)

    class Intermediate(SearchAlgorithm):
        @abstractmethod
        def extra(self) -> None: ...

    assert before == SearchAlgorithm._NAMED

    # An inherited abstract method keeps a subclass abstract too.
    class StillAbstract(Intermediate):
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    assert before == SearchAlgorithm._NAMED


def test_concrete_subclass_registers_by_name_and_needs_one():
    class Named(SearchAlgorithm):
        name = "test_named_search"

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    try:
        assert SearchAlgorithm._NAMED["test_named_search"] is Named
        with pytest.raises(TypeError, match="must name itself"):

            class Unnamed(SearchAlgorithm):
                def __call__(self, *args, **kwargs):
                    raise NotImplementedError

    finally:
        SearchAlgorithm._NAMED.pop("test_named_search", None)
