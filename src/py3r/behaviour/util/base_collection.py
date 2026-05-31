# src/py3r/behaviour/util/base_collection.py

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import MutableMapping
from typing import Any, Self

import pandas as pd

from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import BatchResult
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    read_manifest,
    write_manifest,
)


class _EachProxy:
    """
    Dynamic batch facade exposed as ``collection.each``.

    This proxy always uses smart argument dispatch:
    - ``BatchResult`` values are treated as per-handle maps only when keys
      match exactly (flat handle keys, or grouped shape keys), and
    - everything else is broadcast as a scalar value to all leaves.
    """

    def __init__(self, parent: BaseCollection, *, force_batch: bool = False):
        self._parent = parent
        self._force_batch = force_batch
        self._forcebatch_proxy: _EachProxy | None = None

    def __getattr__(self, name: str):
        leaf_attr = self._parent._get_leaf_callable(name)

        def _batch_wrapper(*args, **kwargs):
            # Smart dispatch: exact-key mappings are mapped per-handle;
            # all other args/kwargs are scalar-broadcast.
            result = self._parent._invoke_batch_mapped(name, args=args, kwargs=kwargs)
            if self._force_batch:
                return result
            return self._parent._maybe_upcast_batch_result(result)

        # Preserve hover docs/name as best effort for dynamic wrappers.
        try:
            _batch_wrapper.__name__ = name  # type: ignore[attr-defined]
            _batch_wrapper.__doc__ = getattr(leaf_attr, "__doc__", None)  # type: ignore[attr-defined]
        except Exception:
            pass
        return _batch_wrapper

    @property
    def forcebatch(self):
        """Return a proxy that always returns BatchResult."""
        if self._forcebatch_proxy is None:
            self._forcebatch_proxy = _EachProxy(
                self._parent,
                force_batch=True,
            )
        return self._forcebatch_proxy

    def __dir__(self):
        base = set(super().__dir__())
        try:
            flat_parent = self._parent.flatten()
            example_leaf = next(iter(flat_parent.values()))
        except StopIteration:
            return sorted(base)
        names = set()
        for n in dir(example_leaf):
            if n.startswith("_"):
                continue
            try:
                # Avoid invoking descriptors/properties during completion.
                attr = inspect.getattr_static(example_leaf, n)
            except Exception:
                continue
            if callable(attr):
                names.add(n)
        return sorted(base | names)


class BaseCollection(MutableMapping):
    """
    Abstract base class for collections of objects (e.g., Features, Tracking, Summary).
    Provides groupby and flatten logic, and basic dict-like access.
    Subclasses must define:
        - _element_type: the type of elements (e.g., Features)
        - _multiple_collection_type: the MultipleCollection class to return from groupby
        - from_list(cls, objs): classmethod to construct from a list of elements.

    Examples
    --------
    A concrete example using TrackingCollection:

    ```pycon
    >>> import tempfile, shutil
    >>> from pathlib import Path
    >>> from py3r.behaviour.util.docdata import data_path
    >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    >>> from py3r.behaviour.tracking.tracking import Tracking
    >>> with tempfile.TemporaryDirectory() as d:
    ...     d = Path(d)
    ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...         a = d / 'A.csv'; b = d / 'B.csv'
    ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
    ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
    >>> list(sorted(coll.keys()))
    ['A', 'B']
    >>> len(coll)
    2

    ```
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_element_type"):
            raise TypeError(
                f"{cls.__name__} must define '_element_type' as a class attribute "
                f"(e.g. '_element_type = MyClass')."
            )

    def __init__(self, obj_dict):
        self._obj_dict = dict(obj_dict)  # {handle: element or sub-collection}
        self._groupby_tags = None
        self._each_proxy: _EachProxy | None = None
        self._each_forcebatch_proxy: _EachProxy | None = None

    def _batch_error_context(self, key):
        # If this is a grouped view, treat top-level keys as collection names
        if self.is_grouped:
            return dict(collection_name=key, object_name=None)
        # Default: flat collection (key refers to object name/handle)
        return dict(collection_name=None, object_name=key)

    def _invoke_batch(self, _method_name: str, *args, **kwargs) -> BatchResult:
        """
        Group-aware batch dispatcher for leaf methods (fail-fast).

        This dispatcher does uniform broadcast only: every leaf receives the
        same ``args``/``kwargs`` values unchanged.

        Applies the named method to each leaf object. If any leaf raises, a
        BatchProcessError is raised immediately. On complete success, returns a
        BatchResult of leaf return values. When grouped, produces a nested
        mapping of group -> BatchResult.
        """
        results = {}
        if getattr(self, "is_grouped", False):
            for group_key, subcoll in self.items():
                group_results = {}
                for obj_key, obj in subcoll.items():
                    try:
                        group_results[obj_key] = getattr(obj, _method_name)(*args, **kwargs)
                    except Exception as e:
                        raise BatchProcessError(
                            collection_name=group_key,
                            object_name=obj_key,
                            method=_method_name,
                            original_exception=e,
                        ) from e
                results[group_key] = BatchResult(group_results, subcoll)
        else:
            for key, obj in self.items():
                try:
                    results[key] = getattr(obj, _method_name)(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=None,
                        object_name=key,
                        method=_method_name,
                        original_exception=e,
                    ) from e
        return BatchResult(results, self)

    def _invoke_batch_mapped(
        self,
        _method_name: str,
        *,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> BatchResult:
        """
        Smart batch dispatcher (fail-fast) used by ``collection.each``.

        For every positional/keyword argument, this method decides between
        per-handle mapping and scalar broadcast:

        - Non-mapping values are always scalar-broadcast.
        - Only ``BatchResult`` values are eligible for mapped semantics.
        - A ``BatchResult`` is treated as mapped only when it matches exactly:
          - flat map: ``{handle: value}`` with keys equal to flattened handles
          - grouped map: ``{group: {handle: value}}`` with exact current group
            keys and exact nested handle keys for each group.
        - For grouped collections, a grouped-shaped ``BatchResult`` from a
          different grouping layout may still be accepted if it can be
          flattened unambiguously to a complete flat handle map; in that case
          a warning is emitted and handle-based mapping is used.
        - If a ``BatchResult`` cannot be resolved to one of the above mapping
          shapes, an error is raised.
        - Plain ``dict`` values are scalar-broadcast unless at least one value
          is a ``BatchResult``. In that mixed case, each embedded
          ``BatchResult`` is mapped per leaf while non-``BatchResult`` entries
          are broadcast unchanged.

        If any leaf raises, a ``BatchProcessError`` is raised immediately.
        On complete success, returns a ``BatchResult`` of leaf return values
        (nested by group when grouped).
        """
        if kwargs is None:
            kwargs = {}

        from collections.abc import Mapping

        grouped = getattr(self, "is_grouped", False)
        group_keys = tuple(self.keys()) if grouped else tuple()
        group_key_set = set(group_keys)
        flat_keys = tuple(self.flatten().keys()) if grouped else tuple(self.keys())
        flat_key_set = set(flat_keys)
        grouped_handle_sets = {gk: set(self[gk].keys()) for gk in group_keys} if grouped else {}
        mode_cache: dict[int, str] = {}
        value_cache: dict[int, object] = {}

        def _flatten_grouped_map(spec: Mapping) -> dict | None:
            """Flatten {group: {handle: value}} to {handle: value} if unambiguous."""
            flat = {}
            for group_val in spec.values():
                if not isinstance(group_val, Mapping):
                    return None
                for handle, v in group_val.items():
                    if handle in flat:
                        return None
                    flat[handle] = v
            return flat

        def _resolve_mode(spec):
            # Only BatchResult supports mapped argument semantics.
            if not isinstance(spec, BatchResult):
                return "scalar", spec

            spec_id = id(spec)
            if spec_id in mode_cache:
                return mode_cache[spec_id], value_cache[spec_id]

            keys = set(spec.keys())

            # Handle map works regardless of grouped/flat current view.
            if keys == flat_key_set:
                mode, value = "flat", spec
            elif grouped and keys == group_key_set:
                # Current grouped shape map: require exact nested handle keys.
                nested_ok = all(
                    isinstance(spec[gk], Mapping)
                    and set(spec[gk].keys()) == grouped_handle_sets[gk]
                    for gk in group_keys
                )
                if nested_ok:
                    mode, value = "grouped", spec
                else:
                    flattened = _flatten_grouped_map(spec)
                    if flattened is not None and set(flattened.keys()) == flat_key_set:
                        warnings.warn(
                            "Mapped argument uses grouped keys that"
                            "do not match the current grouping; "
                            "falling back to handle-based mapping.",
                            stacklevel=3,
                        )
                        mode, value = "flat", flattened
                    else:
                        raise KeyError(
                            "BatchResult mapping keys do not match current grouped structure "
                            "or flattened handle keys."
                        )
            else:
                # Allow grouped-structured maps from a previous grouping by flattening by handle.
                flattened = _flatten_grouped_map(spec)
                if grouped and flattened is not None and set(flattened.keys()) == flat_key_set:
                    warnings.warn(
                        "Mapped argument grouping differs from the current grouped view; "
                        "falling back to handle-based mapping.",
                        stacklevel=3,
                    )
                    mode, value = "flat", flattened
                else:
                    raise KeyError(
                        "BatchResult mapping keys do not match flattened collection handle keys."
                    )

            mode_cache[spec_id] = mode
            value_cache[spec_id] = value
            return mode, value

        def select(spec, group_key, obj_key):
            if isinstance(spec, Mapping) and not isinstance(spec, BatchResult):
                has_embedded_batch = any(isinstance(v, BatchResult) for v in spec.values())
                if has_embedded_batch:
                    return {k: select(v, group_key, obj_key) for k, v in spec.items()}
            mode, resolved = _resolve_mode(spec)
            if mode == "grouped":
                return resolved[group_key][obj_key]
            if mode == "flat":
                return resolved[obj_key]
            return resolved

        results = {}
        if grouped:
            for group_key, subcoll in self.items():
                group_results = {}
                for obj_key, obj in subcoll.items():
                    try:
                        leaf_args = tuple(select(a, group_key, obj_key) for a in args)
                        leaf_kwargs = {k: select(v, group_key, obj_key) for k, v in kwargs.items()}
                        group_results[obj_key] = getattr(obj, _method_name)(
                            *leaf_args, **leaf_kwargs
                        )
                    except Exception as e:
                        raise BatchProcessError(
                            collection_name=group_key,
                            object_name=obj_key,
                            method=_method_name,
                            original_exception=e,
                        ) from e
                results[group_key] = BatchResult(group_results, subcoll)
        else:
            for obj_key, obj in self.items():
                try:
                    leaf_args = tuple(select(a, None, obj_key) for a in args)
                    leaf_kwargs = {k: select(v, None, obj_key) for k, v in kwargs.items()}
                    results[obj_key] = getattr(obj, _method_name)(*leaf_args, **leaf_kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=None,
                        object_name=obj_key,
                        method=_method_name,
                        original_exception=e,
                    ) from e
        return BatchResult(results, self)

    def __getitem__(self, key):
        """
        Get element by handle (str), by integer index, or by slice.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> isinstance(coll['A'].data, type(coll['B'].data))
        True
        >>> isinstance(coll[0].data, type(coll['A'].data))
        True
        >>> isinstance(coll[0:1], type(coll))
        True

        ```
        """
        if isinstance(key, int):
            handle = list(self._obj_dict)[key]
            return self._obj_dict[handle]
        elif isinstance(key, slice):
            handles = list(self._obj_dict)[key]
            return self.__class__({h: self._obj_dict[h] for h in handles})
        else:
            return self._obj_dict[key]

    def __setitem__(self, key, value):
        element_cls = getattr(self, "_element_type", None)
        if element_cls is not None and not isinstance(value, element_cls):
            raise TypeError(f"Value must be a {element_cls.__name__}, got {type(value).__name__}")
        cn = self.__class__.__name__
        warnings.warn(
            f"Direct assignment to {cn} is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._obj_dict[key] = value

    def __delitem__(self, key):
        del self._obj_dict[key]

    def __iter__(self):
        return iter(self._obj_dict)

    def __len__(self):
        """
        Number of elements (or groups if grouped).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> len(coll)
        2

        ```
        """
        return len(self._obj_dict)

    def values(self):
        """
        Values iterator (elements or sub-collections).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> len(list(coll.values())) == 2
        True

        ```
        """
        return self._obj_dict.values()

    def items(self):
        """
        Items iterator (handle, element).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> sorted([h for h, _ in coll.items()])
        ['A', 'B']

        ```
        """
        return self._obj_dict.items()

    def keys(self):
        """
        Keys iterator (handles or group keys).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> list(sorted(coll.keys()))
        ['A', 'B']

        ```
        """
        return self._obj_dict.keys()

    def _get_leaf_callable(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

        flat_self = self.flatten()
        try:
            example_leaf = next(iter(flat_self.values()))
        except StopIteration:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}") from None

        leaf_attr = getattr(example_leaf, name, None)
        if not callable(leaf_attr):
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")
        return leaf_attr

    def _maybe_upcast_batch_result(self, result):
        """
        Convert a BatchResult into a collection when all leaves are element type.
        Otherwise return the result unchanged.
        """
        if not isinstance(result, BatchResult):
            return result

        element_type = getattr(self, "_element_type", None)
        if element_type is None:
            return result

        if getattr(self, "is_grouped", False):
            if set(result.keys()) != set(self.keys()):
                return result
            grouped_new = {}
            for gkey, subres in result.items():
                if not isinstance(subres, dict):
                    return result
                subcoll = self[gkey]
                if set(subres.keys()) != set(subcoll.keys()):
                    return result
                if not all(isinstance(v, element_type) for v in subres.values()):
                    return result
                grouped_new[gkey] = subcoll.__class__(dict(subres))
            out = self.__class__(grouped_new)
            out._groupby_tags = list(self._groupby_tags) if self._groupby_tags else None
            return out

        if set(result.keys()) != set(self.keys()):
            return result
        if not all(isinstance(v, element_type) for v in result.values()):
            return result
        return self.__class__(dict(result))

    @property
    def each(self):
        """
        Explicit leaf-batch facade with smart argument dispatch.

        ``collection.each.method(...)`` dispatches ``method`` to every leaf and
        uses smart argument handling:
        - exact-key ``BatchResult`` mappings are applied per handle/group-handle
          structure,
        - plain ``dict`` values are broadcast as scalars unless they contain at
          least one embedded ``BatchResult`` value; in that mixed case embedded
          ``BatchResult`` values are mapped per leaf and other dict values are
          broadcast unchanged.

        Returns either:
        - a collection (when all leaf returns are collection element type), or
        - a BatchResult otherwise.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.collection_utils import BatchResult
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> def _mk(handle):
        ...     df = pd.DataFrame({"bp.x":[0.0,1.0], "bp.y":[0.0,1.0]}, index=[0,1])
        ...     t = Tracking(
        ...         df,
        ...         {"fps": 30.0, "rescale_distance_method": "dummy"},
        ...         handle=handle,
        ...     )
        ...     return Features(t)
        >>> fc = FeaturesCollection({"A": _mk("A"), "B": _mk("B")})
        >>> food = BatchResult({
        ...     "A": pd.Series([True, False], index=[0,1]),
        ...     "B": pd.Series([False, True], index=[0,1]),
        ... }, fc)
        >>> corner = pd.Series([False, False], index=[0,1])
        >>> out = fc.each.compose_state_from_booleans({"food": food, "corner": corner})
        >>> isinstance(out, BatchResult)
        True

        ```
        """
        if self._each_proxy is None:
            self._each_proxy = _EachProxy(self)
        return self._each_proxy

    @property
    def each_forcebatch(self):
        """
        Explicit leaf-batch facade that always returns ``BatchResult``.

        Dispatch semantics match ``each`` (smart mapping/broadcast); only the
        return-shape behavior differs.
        """
        if self._each_forcebatch_proxy is None:
            self._each_forcebatch_proxy = _EachProxy(self, force_batch=True)
        return self._each_forcebatch_proxy

    # ---- Dynamic fallback for user-extended leaf APIs ----
    def __getattr__(self, name):
        """
        Hard-fail direct batch passthrough and point users to ``.each``.

        If ``name`` resolves to a public callable on leaf objects, return a
        callable that raises a migration error when invoked:
        ``collection.method(...)`` -> ``collection.each.method(...)``.
        Private/dunder names are not intercepted.
        """
        leaf_attr = self._get_leaf_callable(name)

        # Build a thin wrapper that raises with migration guidance.
        def _batch_wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"Direct batch passthrough via {self.__class__.__name__}.{name}() "
                f"was removed; use {self.__class__.__name__}.each.{name}() instead."
            )

        # Best-effort attach docstring/name for nicer help() / hover info
        try:
            _batch_wrapper.__name__ = name  # type: ignore[attr-defined]
            _batch_wrapper.__doc__ = getattr(leaf_attr, "__doc__", None)  # type: ignore[attr-defined]
        except Exception:
            pass
        return _batch_wrapper

    @classmethod
    def from_list(cls, objs):
        """
        Construct a collection from a list of items, using their .handle as the key.
        Raises a clear error if any item does not have a .handle attribute.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='A', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='B', fps=30)
        >>> coll = TrackingCollection.from_list([t1, t2])
        >>> list(sorted(coll.keys()))
        ['A', 'B']

        ```
        """
        try:
            obj_dict = {obj.handle: obj for obj in objs}
        except AttributeError as e:
            raise TypeError(
                f"All items must have a .handle attribute to use "
                f"{cls.__name__}.from_list(). "
                "This method is only for flat collections of individual items."
            ) from e
        return cls(obj_dict)

    @classmethod
    def merge(cls, collections: list[Self], *, copy: bool = False) -> Self:
        """
        Merge multiple collections into a single flat collection containing
        all leaf elements from each input.

        Each input collection is flattened before merging, so grouped inputs
        are supported. The result is always a new flat collection. Leaves are
        shared by reference unless ``copy=True``.

        Args:
            collections: Two or more collections of the same concrete type. Every
                element across all collections must have a unique handle.
            copy: If True, each leaf is copied (via its ``.copy()`` method) so that
                the merged collection is fully independent of the originals.

        Returns:
            A new flat collection containing all leaves.

        Raises:
            ValueError: If *collections* is empty, or if any handles are duplicated.
            TypeError: If any input is not an instance of the calling class.

        Warns:
            UserWarning: If the tag key sets differ across input collections (the
                merged collection will have mixed tag coverage).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...         _ = shutil.copy(p, d / 'C.csv'); _ = shutil.copy(p, d / 'D.csv')
        ...     c1 = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        ...     c2 = TrackingCollection.from_dlc({'C': str(d/'C.csv'), 'D': str(d/'D.csv')}, fps=30)
        >>> merged = TrackingCollection.merge([c1, c2])
        >>> sorted(merged.keys())
        ['A', 'B', 'C', 'D']
        >>> len(merged)
        4

        ```
        """
        if not collections:
            raise ValueError("merge() requires at least one collection.")

        for i, coll in enumerate(collections):
            if not isinstance(coll, cls):
                raise TypeError(
                    f"All collections must be {cls.__name__} instances, "
                    f"but item {i} is {type(coll).__name__}."
                )

        # Flatten all inputs and gather leaves + tag schemas per collection
        all_leaves = []
        tag_key_sets = []
        for coll in collections:
            flat = coll.flatten()
            leaves = list(flat.values())
            all_leaves.extend(leaves)
            coll_tags = set()
            for obj in leaves:
                tags = getattr(obj, "tags", None)
                if isinstance(tags, dict):
                    coll_tags.update(tags.keys())
            tag_key_sets.append(coll_tags)

        # Check for handle collisions
        seen = set()
        duplicates = []
        for obj in all_leaves:
            h = getattr(obj, "handle", id(obj))
            if h in seen:
                duplicates.append(h)
            seen.add(h)
        if duplicates:
            unique_dupes = sorted(set(str(d) for d in duplicates))
            raise ValueError(
                f"Cannot merge: duplicate handles found across collections: "
                f"{unique_dupes}. Each element must have a unique handle."
            )

        # Warn about mismatched tag schemas across collections
        if len(tag_key_sets) > 1:
            all_tag_keys = set.union(*tag_key_sets)
            if all_tag_keys and not all(ts == all_tag_keys for ts in tag_key_sets):
                per_coll = ", ".join(
                    f"collection {i}: {{{', '.join(sorted(ts))}}}"
                    if ts
                    else f"collection {i}: (none)"
                    for i, ts in enumerate(tag_key_sets)
                )
                warnings.warn(
                    f"Merging collections with different tag schemas. "
                    f"The merged collection will have mixed tag coverage. "
                    f"Tag keys per collection — {per_coll}",
                    stacklevel=2,
                )

        # Optionally deep-copy leaves for full independence from originals
        if copy:
            copied = []
            for obj in all_leaves:
                try:
                    copied.append(obj.copy())
                except AttributeError:
                    raise NotImplementedError(
                        f"{type(obj).__name__} does not implement copy()"
                    ) from None
            all_leaves = copied

        return cls.from_list(all_leaves)

    def groupby(self, tags):
        """
        Group the collection by one or more existing tag names.
        Returns a grouped view (this same collection type) whose values are
        sub-collections keyed by a tuple of tag values in the order provided.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('group','G1'); coll['B'].add_tag('group','G2')
        >>> g = coll.groupby('group')
        >>> g.is_grouped
        True
        >>> sorted(g.group_keys)
        [('G1',), ('G2',)]

        ```
        """
        flat_self = self.flatten()

        if isinstance(tags, str):
            tags = [tags]
        tags = list(tags)
        groups = {}
        missing = []
        for obj in flat_self.values():
            try:
                key = tuple(str(obj.tags[tag]) for tag in tags)
            except KeyError as e:
                missing.append((getattr(obj, "handle", None), e.args[0]))
                continue
            groups.setdefault(key, []).append(obj)
        if missing:
            missing_str = "\n".join(f"{handle}: {tag}" for handle, tag in missing)
            raise ValueError(f"The following elements are missing required tags:\n{missing_str}")

        group_collections = {key: self.__class__.from_list(objs) for key, objs in groups.items()}
        grouped = self.__class__(group_collections)
        grouped._groupby_tags = tags
        return grouped

    def flatten(self):
        """
        Flatten a MultipleCollection to a flat Collection.
        If already flat, return self.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('group','G1'); coll['B'].add_tag('group','G1')
        ...     g = coll.groupby('group')
        >>> flat = g.flatten()
        >>> flat.is_grouped
        False
        >>> sorted(flat.keys())
        ['A', 'B']

        ```
        """
        # If empty, just return self
        if not self._obj_dict:
            return self

        first_value = next(iter(self._obj_dict.values()))
        # If the first value is not a sub-collection (i.e., is a leaf), return self
        if not isinstance(first_value, BaseCollection):
            return self

        # Otherwise, flatten
        all_objs = []
        for obj in self.values():
            if isinstance(obj, BaseCollection):
                all_objs.extend(obj.values())
            else:
                all_objs.append(obj)
        flat_cls = type(first_value)
        return flat_cls.from_list(all_objs)

    def __repr__(self):
        cn = self.__class__.__name__
        if self.is_grouped:
            return f"<{cn} grouped by {self._groupby_tags} with {len(self)} groups>"
        return f"<{cn} with {len(self)} {self._element_type.__name__} objects>"

    # ---- Grouped view helpers ----
    @property
    def is_grouped(self):
        """
        True if this collection is a grouped view.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> coll.is_grouped
        False

        ```
        """
        if not self._obj_dict:
            return False
        return isinstance(next(iter(self._obj_dict.values())), BaseCollection)

    @property
    def groupby_tags(self):
        """The tag names used to form this grouped view (or None if flat)."""
        return getattr(self, "_groupby_tags", None)

    @property
    def group_keys(self):
        """
        Keys for the groups in a grouped view. Empty list if not grouped.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('group','G1'); coll['B'].add_tag('group','G2')
        >>> g = coll.groupby('group')
        >>> sorted(g.group_keys)
        [('G1',), ('G2',)]

        ```
        """
        if not self.is_grouped:
            return []
        return list(self._obj_dict.keys())

    def get_group(self, key):
        """
        Get a sub-collection by group key from a grouped view.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('group','G1'); coll['B'].add_tag('group','G2')
        >>> g = coll.groupby('group')
        >>> sub = g.get_group(('G1',))
        >>> list(sub.keys())
        ['A']

        ```
        """
        if not self.is_grouped:
            raise ValueError("Collection is not grouped.")
        return self._obj_dict[key]

    def regroup(self):
        """
        Recompute the same grouping using the current tags and the original
        grouping tag order. If not grouped, returns self.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('group','G1'); coll['B'].add_tag('group','G1')
        ...     g = coll.groupby('group')
        ...     coll['B'].add_tag('group','G2', overwrite=True)  # change tag
        >>> g2 = g.regroup()
        >>> sorted(g2.group_keys)
        [('G1',), ('G2',)]

        ```
        """
        if not self.is_grouped or not self._groupby_tags:
            return self
        return self.flatten().groupby(self._groupby_tags)

    # ---- Transform helpers ----
    def tags_info(
        self,
        *,
        include_value_counts: bool = False,
    ) -> pd.DataFrame:
        """
        Summarize tag presence across the collection's leaf objects.
        Works for flat and grouped collections. If `include_value_counts` is True,
        include a column 'value_counts' with a dict of `value->count` for each tag.
        Returns a `pandas.DataFrame` with columns:
        `['tag', 'attached_to', 'missing_from', 'unique_values', ('value_counts')]`.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     coll['A'].add_tag('genotype', 'WT')
        ...     coll['B'].add_tag('timepoint', 'T1')
        >>> info = coll.tags_info(include_value_counts=True)
        >>> int(info.loc['genotype','attached_to'])
        1
        >>> int(info.loc['genotype','missing_from'])
        1
        >>> int(info.loc['genotype','unique_values'])
        1
        >>> info.loc['genotype','value_counts']
        {'WT': 1}
        >>> int(info.loc['timepoint','attached_to'])
        1

        ```
        """

        def summarize_leaves(leaves: list):
            # Collect all tag keys and their values across leaves
            all_keys = set()
            values_by_key: dict[str, list] = {}
            total = len(leaves)
            for obj in leaves:
                tags = getattr(obj, "tags", None)
                if not isinstance(tags, dict):
                    continue
                for k, v in tags.items():
                    all_keys.add(k)
                    values_by_key.setdefault(k, []).append(v)
            records = []
            for k in sorted(all_keys):
                vals = values_by_key.get(k, [])
                present = len(vals)
                missing = total - present
                unique_values = len(set(vals)) if present else 0
                rec = {
                    "tag": k,
                    "attached_to": present,
                    "missing_from": missing,
                    "unique_values": unique_values,
                }
                if include_value_counts:
                    # preserve simple dict for readability
                    vc = pd.Series(vals, dtype="object").value_counts(dropna=False)
                    rec["value_counts"] = {str(idx): int(cnt) for idx, cnt in vc.items()}
                records.append(rec)
            if not records:
                # No tags present anywhere; return empty frame with expected columns
                cols = ["tag", "attached_to", "missing_from", "unique_values"]
                if include_value_counts:
                    cols.append("value_counts")
                return pd.DataFrame(columns=cols).set_index("tag")
            df = pd.DataFrame.from_records(records).set_index("tag")
            # Ensure integer dtype where possible
            for c in ["attached_to", "missing_from", "unique_values"]:
                if c in df:
                    df[c] = df[c].astype("int64")
            return df

        # Flat or aggregate across groups
        leaves = list(self.flatten().values())
        return summarize_leaves(leaves)

    def map_leaves(self, fn):
        """
        Apply a function to every leaf element and return a new collection of the
        same type. Preserves grouping shape and groupby metadata when grouped.

        fn: callable(Element) -> ElementLike

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> sub = coll.map_leaves(lambda t: t.loc[0:1])
        >>> all(len(t.data) == 2 for t in sub.values())
        True

        ```
        """
        if self.is_grouped:
            grouped_new = {}
            for gkey, sub in self.items():
                # sub is a flat collection (same class as self), map each leaf
                new_sub_dict = {handle: fn(obj) for handle, obj in sub.items()}
                grouped_new[gkey] = sub.__class__(new_sub_dict)
            out = self.__class__(grouped_new)
            out._groupby_tags = list(self._groupby_tags) if self._groupby_tags else None
            return out
        # Flat case
        new_dict = {handle: fn(obj) for handle, obj in self.items()}
        return self.__class__(new_dict)

    def copy(self):
        """
        Creates a copy of the BaseCollection.
        Raises NotImplementedError if any leaf does not implement copy().

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv')
        ...         _ = shutil.copy(p, d / 'B.csv')
        ...     coll = TrackingCollection.from_folder(
        ...         str(d), tracking_loader=Tracking.from_dlc, fps=30
        ...     )
        >>> coll_copy = coll.copy()
        >>> sorted(coll_copy.keys())
        ['A', 'B']

        ```
        """

        def _copy_leaf(t):
            try:
                return t.copy()
            except AttributeError:
                raise NotImplementedError(f"{type(t).__name__} does not implement copy()") from None

        return self.map_leaves(_copy_leaf)

    # ---- Generic persistence for collections ----
    def save(self, dirpath: str, *, overwrite: bool = False, data_format: str = "parquet") -> None:
        """
        Save this collection to a directory. Preserves grouping and delegates to
        leaf objects' save(dirpath, data_format, overwrite=True).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil, os
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     out = d / 'coll'
        ...     coll.save(str(out), overwrite=True, data_format='csv')
        ...     # collection-level manifest at top-level
        ...     assert os.path.exists(os.path.join(str(out), 'manifest.json'))
        ...     # element-level manifests under elements/<handle>/
        ...     el_manifest = os.path.join(str(out), 'elements', 'A', 'manifest.json')
        ...     assert os.path.exists(el_manifest)

        ```
        """
        target = begin_save(dirpath, overwrite)
        is_grouped = getattr(self, "is_grouped", False)
        manifest: dict = {
            "schema_version": SchemaVersion,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "is_grouped": is_grouped,
            "groupby_tags": getattr(self, "groupby_tags", None),
            "elements_index": {},
        }
        if is_grouped:
            for gkey, sub in self.items():
                subdir = os.path.join("groups", str(gkey))
                abs_subdir = os.path.join(target, subdir)
                os.makedirs(abs_subdir, exist_ok=True)
                manifest["elements_index"][str(gkey)] = {}
                for handle, obj in sub.items():
                    leaf_dir_rel = os.path.join(subdir, handle)
                    leaf_dir_abs = os.path.join(target, leaf_dir_rel)
                    # delegate to leaf
                    if hasattr(obj, "save"):
                        obj.save(leaf_dir_abs, data_format=data_format, overwrite=True)
                    else:
                        raise AttributeError(f"Leaf object {type(obj)} has no save()")
                    manifest["elements_index"][str(gkey)][handle] = leaf_dir_rel
        else:
            elems_dir = os.path.join(target, "elements")
            os.makedirs(elems_dir, exist_ok=True)
            for handle, obj in self.items():
                leaf_dir_rel = os.path.join("elements", handle)
                leaf_dir_abs = os.path.join(target, leaf_dir_rel)
                if hasattr(obj, "save"):
                    obj.save(leaf_dir_abs, data_format=data_format, overwrite=True)
                else:
                    raise AttributeError(f"Leaf object {type(obj)} has no save()")
                manifest["elements_index"][handle] = leaf_dir_rel
        write_manifest(target, manifest)

    @classmethod
    def load(cls, dirpath: str):
        """
        Load a collection previously saved with save(). Uses the class's
        _element_type.load to reconstruct leaves.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     out = d / 'coll'
        ...     coll.save(str(out), overwrite=True, data_format='csv')
        ...     coll2 = TrackingCollection.load(str(out))
        >>> list(sorted(coll2.keys()))
        ['A', 'B']

        ```
        """
        manifest = read_manifest(dirpath)
        is_grouped = manifest.get("is_grouped", False)
        index = manifest.get("elements_index", {})
        try:
            element_cls = cls._element_type
        except AttributeError:
            raise TypeError(
                f"{cls.__name__} must define _element_type to load() collections"
            ) from None
        if not hasattr(element_cls, "load"):
            raise TypeError(f"{element_cls} must implement classmethod load(dirpath)")
        if is_grouped:
            grouped = {}
            for gkey, mapping in index.items():
                sub = {}
                for handle, rel in mapping.items():
                    sub[handle] = element_cls.load(os.path.join(dirpath, rel))
                grouped[gkey] = cls(sub)
            out = cls(grouped)
            out._groupby_tags = manifest.get("groupby_tags")
            return out
        else:
            flat = {
                handle: element_cls.load(os.path.join(dirpath, rel))
                for handle, rel in index.items()
            }
            return cls(flat)
