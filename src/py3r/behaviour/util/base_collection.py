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
        self._obj_dict = dict(obj_dict)  # {handle: element}

    def _batch_error_context(self, key):
        # Default: flat collection
        return dict(collection_name=None, object_name=key)

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self._obj_dict.items():
                try:
                    method = getattr(obj, name)
                    results[key] = method(*args, **kwargs)
                except Exception as e:
                    ctx = self._batch_error_context(key)
                    raise BatchProcessError(
                        collection_name=ctx["collection_name"],
                        object_name=ctx["object_name"],
                        method=getattr(e, "method", name),
                        original_exception=getattr(e, "original_exception", e),
                    ) from e
            return BatchResult(results, self)

        return batch_method

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

    def _batch_error_context(self, key):
        # Default: flat collection
        return dict(collection_name=None, object_name=key)

    def _resolve_multiple_collection_type(self):
        multiple_cls = self._multiple_collection_type
        if isinstance(multiple_cls, str):
            # Late import based on string name
            if multiple_cls == "MultipleTrackingCollection":
                from py3r.behaviour.tracking.multiple_tracking_collection import (
                    MultipleTrackingCollection,
                )

                multiple_cls = MultipleTrackingCollection
            elif multiple_cls == "MultipleFeaturesCollection":
                from py3r.behaviour.features.multiple_features_collection import (
                    MultipleFeaturesCollection,
                )

                multiple_cls = MultipleFeaturesCollection
            elif multiple_cls == "MultipleSummaryCollection":
                from py3r.behaviour.summary.multiple_summary_collection import (
                    MultipleSummaryCollection,
                )

                multiple_cls = MultipleSummaryCollection
            self._multiple_collection_type = multiple_cls  # cache for next time
        return multiple_cls

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
        Group the collection by one or more tags.
        Returns a MultipleCollection object with group names as keys.
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

        def group_name(key_tuple):
            return "_".join(str(v) for v in key_tuple)

        group_collections = {
            group_name(key): self.__class__.from_list(objs)
            for key, objs in groups.items()
        }
        multiple_cls = self._resolve_multiple_collection_type()
        return multiple_cls(group_collections)

    def flatten(self):
        """
        If this is a MultipleCollection, flatten to a single Collection.
        If already flat, return self.
        """
        all_objs = []
        for obj in self.values():
            if isinstance(obj, self.__class__):
                all_objs.extend(obj.values())
            else:
                all_objs.append(obj)
        flat_cls = type(all_objs[0])
        return flat_cls.from_list(all_objs)

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} {self._element_type.__name__} objects>"


class BaseMultipleCollection(BaseCollection):
    def _batch_error_context(self, key):
        # Multiple collection: key is the group name
        return dict(collection_name=key, object_name=None)
