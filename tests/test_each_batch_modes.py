from __future__ import annotations

import pytest

from py3r.behaviour.exceptions import BatchProcessError
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

    def label(self, value: str) -> str:
        return f"{self.handle}:{value}"

    def passthrough(self, value):
        return value


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


def test_each_supports_batchresult_per_handle_positional_args():
    coll = _make_collection()
    mapped = BatchResult({"A": "left", "B": "right"}, coll)
    out = coll.each.label(mapped)
    assert isinstance(out, BatchResult)
    assert out["A"] == "A:left"
    assert out["B"] == "B:right"


def test_each_broadcasts_scalar_args():
    coll = _make_collection()
    out = coll.each.label("same")
    assert isinstance(out, BatchResult)
    assert out["A"] == "A:same"
    assert out["B"] == "B:same"


def test_each_forcebatch_keeps_batch_result_with_smart_dispatch():
    coll = _make_collection()
    out = coll.each.forcebatch.touch(inplace=False)
    assert isinstance(out, BatchResult)
    assert all(isinstance(v, DummyLeaf) for v in out.values())


def test_grouped_each_accepts_flat_handle_batchresult_mapping():
    grouped = _make_collection().groupby("group")
    mapped = BatchResult({"A": "left", "B": "right"}, grouped)
    out = grouped.each.label(mapped)
    assert isinstance(out, BatchResult)
    assert out[("g1",)]["A"] == "A:left"
    assert out[("g2",)]["B"] == "B:right"


def test_grouped_each_warns_and_falls_back_when_batchresult_group_keys_differ():
    grouped = _make_collection().groupby("group")
    old_grouped_map = BatchResult(
        {
            ("old_g1",): {"A": "left"},
            ("old_g2",): {"B": "right"},
        },
        grouped,
    )
    with pytest.warns(UserWarning, match="falling back to handle-based mapping"):
        out = grouped.each.label(old_grouped_map)
    assert out[("g1",)]["A"] == "A:left"
    assert out[("g2",)]["B"] == "B:right"


def test_each_non_matching_dict_is_broadcast_not_mapped():
    coll = _make_collection()
    cfg = {"window": 3, "method": "mean"}  # not keyed by handles
    out = coll.each.passthrough(cfg)
    assert isinstance(out, BatchResult)
    assert out["A"] is cfg
    assert out["B"] is cfg


def test_each_invalid_batchresult_mapping_raises():
    coll = _make_collection()
    bad = BatchResult({"A": "left"}, coll)  # missing B
    with pytest.raises(BatchProcessError, match="BatchResult mapping keys"):
        coll.each.label(bad)
