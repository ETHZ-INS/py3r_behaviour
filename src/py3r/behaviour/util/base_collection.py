# src/py3r/behaviour/util/base_collection.py

from collections.abc import MutableMapping


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

    def __getitem__(self, key):
        return self._obj_dict[key]

    def __setitem__(self, key, value):
        if not isinstance(value, self._element_type):
            raise TypeError(
                f"Value must be {self._element_type.__name__}, got {type(value).__name__}"
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
        return self._multiple_collection_type(group_collections)

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
        return self.__class__.from_list(all_objs)

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} {self._element_type.__name__} objects>"
