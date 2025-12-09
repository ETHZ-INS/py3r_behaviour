# src/py3r/behaviour/util/base_collection.py

from __future__ import annotations

from collections.abc import MutableMapping
import os
import warnings
import pandas as pd

from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import BatchResult
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    write_manifest,
    read_manifest,
)


class BaseCollection(MutableMapping):
    """
    Abstract base class for collections of objects (e.g., Features, Tracking, Summary).
    Provides groupby and flatten logic, and basic dict-like access.
    Subclasses must define:
        - _element_type: the type of elements (e.g., Features)
        - _multiple_collection_type: the MultipleCollection class to return from groupby
        - from_list(cls, objs): classmethod to construct from a list of elements

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

    def __init__(self, obj_dict):
        self._obj_dict = dict(obj_dict)  # {handle: element or sub-collection}
        # Grouped-view metadata defaults: always start flat unless explicitly set later
        self._is_grouped = False
        self._groupby_tags = None

    def _batch_error_context(self, key):
        # If this is a grouped view, treat top-level keys as collection names
        if getattr(self, "_is_grouped", False):
            return dict(collection_name=key, object_name=None)
        # Default: flat collection (key refers to object name/handle)
        return dict(collection_name=None, object_name=key)

    def _invoke_batch(self, _method_name: str, *args, **kwargs) -> BatchResult:
        """
        Group-aware batch dispatcher for leaf methods (fail-fast).

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
                        group_results[obj_key] = getattr(obj, _method_name)(
                            *args, **kwargs
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
        Strict batch dispatcher (fail-fast) that accepts positional and keyword arguments
        where any argument may be either:
          - a scalar (applied uniformly), or
          - a mapping whose keys exactly mirror the collection's structure:
            - flat: {handle: value}
            - grouped: {group_key: {handle: value}}

        The mapping shape must exactly match the collection; missing keys raise KeyError.
        If any leaf raises, a BatchProcessError is raised immediately. On complete
        success, returns a BatchResult of leaf return values (or nested results).
        """
        if kwargs is None:
            kwargs = {}

        def select(spec, group_key, obj_key):
            # If spec is mapping-like, pick value using exact keys; otherwise use as-is
            from collections.abc import Mapping

            if isinstance(spec, Mapping):
                if getattr(self, "is_grouped", False):
                    return spec[group_key][obj_key]
                return spec[obj_key]
            return spec

        results = {}
        if getattr(self, "is_grouped", False):
            for group_key, subcoll in self.items():
                group_results = {}
                for obj_key, obj in subcoll.items():
                    try:
                        leaf_args = tuple(select(a, group_key, obj_key) for a in args)
                        leaf_kwargs = {
                            k: select(v, group_key, obj_key) for k, v in kwargs.items()
                        }
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
                    leaf_kwargs = {
                        k: select(v, None, obj_key) for k, v in kwargs.items()
                    }
                    results[obj_key] = getattr(obj, _method_name)(
                        *leaf_args, **leaf_kwargs
                    )
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
        element_cls = type(self[0])
        if not isinstance(value, element_cls):
            raise TypeError(
                f"Value must be a {element_cls.__name__}, got {type(value).__name__}"
            )
        warnings.warn(
            f"Direct assignment to {self.__class__.__name__} is deprecated and may be removed in a future version.",
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

    # ---- Dynamic fallback for user-extended leaf APIs ----
    def __getattr__(self, name):
        """
        Dynamic batch wrapper fallback for methods that are not explicitly
        provided by generated batch mixins.

        - If `name` is a public callable on the leaf objects, return a callable
          that dispatches via the grouped-aware batch dispatcher.
        - Keeps BatchProcessError wrapping semantics from _invoke_batch.
        - Avoids intercepting private/dunder names.
        """
        if name.startswith("_"):
            # Preserve normal AttributeError semantics for private/dunder
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        # Find a representative leaf to introspect available methods,
        # even when this collection is grouped.
        flat_self = self.flatten()
        try:
            example_leaf = next(iter(flat_self.values()))
        except StopIteration:
            # Empty collection: nothing to expose dynamically
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        leaf_attr = getattr(example_leaf, name, None)
        if not callable(leaf_attr):
            raise AttributeError(
                f"{self.__class__.__name__!s} has no attribute {name!r}"
            )

        # Build a thin wrapper that routes to the batch dispatcher.
        def _batch_wrapper(*args, **kwargs):
            return self._invoke_batch(name, *args, **kwargs)

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
                f"All items must have a .handle attribute to use {cls.__name__}.from_list(). "
                "This method is only for flat collections of individual items."
            ) from e
        return cls(obj_dict)

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
            raise ValueError(
                f"The following elements are missing required tags:\n{missing_str}"
            )

        group_collections = {
            key: self.__class__.from_list(objs) for key, objs in groups.items()
        }
        grouped = self.__class__(group_collections)
        grouped._is_grouped = True
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
        # If the first value is not a collection (i.e., is a leaf), return self
        if not hasattr(first_value, "values") or not callable(first_value.values):
            return self

        # Otherwise, flatten
        all_objs = []
        for obj in self.values():
            if hasattr(obj, "values") and callable(obj.values):
                all_objs.extend(obj.values())
            else:
                all_objs.append(obj)
        flat_cls = type(first_value)
        flat = flat_cls.from_list(all_objs)
        # Ensure returned flat collection is not marked grouped
        if hasattr(flat, "_is_grouped"):
            flat._is_grouped = False
            flat._groupby_tags = None
        return flat

    def __repr__(self):
        if getattr(self, "_is_grouped", False):
            return f"<{self.__class__.__name__} grouped by {self._groupby_tags} with {len(self)} groups>"
        return f"<{self.__class__.__name__} with {len(self)} {self._element_type.__name__} objects>"

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
        return getattr(self, "_is_grouped", False)

    @property
    def groupby_tags(self):
        """
        The tag names used to form this grouped view (or None if flat).
        """
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
        `['tag', 'attached_to', 'missing_from', 'unique_values', ('value_counts')]`

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
        >>> info = coll.tags_info()
        >>> int(info.loc['genotype','present'])
        1
        >>> int(info.loc['timepoint','present'])
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
                    rec["value_counts"] = {
                        str(idx): int(cnt) for idx, cnt in vc.items()
                    }
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
            out._is_grouped = True
            out._groupby_tags = list(self._groupby_tags) if self._groupby_tags else None
            return out
        # Flat case
        new_dict = {handle: fn(obj) for handle, obj in self.items()}
        return self.__class__(new_dict)

    # ---- Generic persistence for collections ----
    def save(
        self, dirpath: str, *, overwrite: bool = False, data_format: str = "parquet"
    ) -> None:
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
        ...     assert os.path.exists(os.path.join(str(out), 'elements', 'A', 'manifest.json'))

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
            element_cls = getattr(cls, "_element_type")
        except AttributeError:
            raise TypeError(
                f"{cls.__name__} must define _element_type to load() collections"
            )
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
            out._is_grouped = True
            out._groupby_tags = manifest.get("groupby_tags")
            return out
        else:
            flat = {
                handle: element_cls.load(os.path.join(dirpath, rel))
                for handle, rel in index.items()
            }
            return cls(flat)
