# src/py3r/behaviour/util/base_collection.py

from __future__ import annotations

from collections.abc import MutableMapping
import warnings

from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import BatchResult


class BaseCollection(MutableMapping):
    """
    Abstract base class for collections of objects (e.g., Features, Tracking, Summary).
    Provides groupby and flatten logic, and basic dict-like access.
    Subclasses must define:
        - _element_type: the type of elements (e.g., Features)
        - _multiple_collection_type: the MultipleCollection class to return from groupby
        - from_list(cls, objs): classmethod to construct from a list of elements
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
        Group-aware batch dispatcher for leaf methods.

        Applies the named method to each leaf object, collecting results into a
        BatchResult. When grouped, produces a nested mapping of group -> BatchResult.
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
                        group_results[obj_key] = BatchProcessError(
                            collection_name=group_key,
                            object_name=obj_key,
                            method=_method_name,
                            original_exception=e,
                        )
                results[group_key] = BatchResult(group_results, subcoll)
        else:
            for key, obj in self.items():
                try:
                    results[key] = getattr(obj, _method_name)(*args, **kwargs)
                except Exception as e:
                    results[key] = BatchProcessError(
                        collection_name=None,
                        object_name=key,
                        method=_method_name,
                        original_exception=e,
                    )
        return BatchResult(results, self)

    def __getitem__(self, key):
        """
        Get element by handle (str), by integer index, or by slice.
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
        return len(self._obj_dict)

    def values(self):
        return self._obj_dict.values()

    def items(self):
        return self._obj_dict.items()

    def keys(self):
        return self._obj_dict.keys()

    @classmethod
    def from_list(cls, objs):
        """
        Construct a collection from a list of items, using their .handle as the key.
        Raises a clear error if any item does not have a .handle attribute.
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
        return getattr(self, "_is_grouped", False)

    @property
    def groupby_tags(self):
        return getattr(self, "_groupby_tags", None)

    @property
    def group_keys(self):
        if not self.is_grouped:
            return []
        return list(self._obj_dict.keys())

    def get_group(self, key):
        if not self.is_grouped:
            raise ValueError("Collection is not grouped.")
        return self._obj_dict[key]

    def regroup(self):
        """
        Recompute the same grouping using the current tags and the original
        grouping tag order. If not grouped, returns self.
        """
        if not self.is_grouped or not self._groupby_tags:
            return self
        return self.flatten().groupby(self._groupby_tags)
