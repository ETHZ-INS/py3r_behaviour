# src/py3r/behaviour/util/base_collection.py

from __future__ import annotations

from collections.abc import MutableMapping
import os
import warnings

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

    # ---- Transform helpers ----
    def map_leaves(self, fn):
        """
        Apply a function to every leaf element and return a new collection of the
        same type. Preserves grouping shape and groupby metadata when grouped.

        fn: callable(Element) -> ElementLike

        Example:
            # Turn a TrackingCollection[TrackingMV] into TrackingCollection[Tracking]
            triangulated = tracking_collection.map_leaves(lambda t: t.stereo_triangulate())
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
