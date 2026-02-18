from __future__ import annotations

from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import BatchResult


class DummyLeaf:
    def __init__(self, handle: str, *, tags: dict | None = None):
        self.handle = handle
        self.tags = {} if tags is None else dict(tags)

    def touch(self, *, inplace: bool = True):
        if inplace:
            self.tags["touched"] = True
            return None
        out = DummyLeaf(self.handle, tags=self.tags)
        out.tags["touched"] = True
        return out

    def value(self) -> int:
        return 1


class DummyCollection(BaseCollection):
    _element_type = DummyLeaf


def _make_collection() -> DummyCollection:
    return DummyCollection(
        {
            "A": DummyLeaf("A", tags={"group": "g1"}),
            "B": DummyLeaf("B", tags={"group": "g2"}),
        }
    )


def test_each_auto_upcasts_when_leaf_objects_returned():
    coll = _make_collection()
    out = coll.each.touch(inplace=False)
    assert isinstance(out, DummyCollection)
    assert all(obj.tags.get("touched") is True for obj in out.values())


def test_each_forcebatch_keeps_batch_result():
    coll = _make_collection()
    out = coll.each_forcebatch.touch(inplace=False)
    assert isinstance(out, BatchResult)
    assert all(isinstance(v, DummyLeaf) for v in out.values())


def test_each_keeps_batch_result_for_non_leaf_values():
    coll = _make_collection()
    out = coll.each.value()
    assert isinstance(out, BatchResult)
    assert out["A"] == 1 and out["B"] == 1


def test_grouped_each_auto_upcasts_and_preserves_grouping():
    grouped = _make_collection().groupby("group")
    out = grouped.each.touch(inplace=False)
    assert isinstance(out, DummyCollection)
    assert out.is_grouped is True
    assert out.groupby_tags == ["group"]
    assert set(out.keys()) == set(grouped.keys())
    for _, sub in out.items():
        assert isinstance(sub, DummyCollection)
        assert all(obj.tags.get("touched") is True for obj in sub.values())
