class _Indexer:
    def __init__(self, parent, slicer):
        self.parent = parent
        self.slicer = slicer

    def __getitem__(self, idx):
        return self.slicer(idx)


class BatchResult(dict):
    def __init__(self, data, parent_collection):
        super().__init__(data)
        self._parent_collection = parent_collection

    def plot(self, *args, **kwargs):
        return self._parent_collection.plot(self, *args, **kwargs)

    def store(self, *args, **kwargs):
        return self._parent_collection.store(self, *args, **kwargs)

    # ----- Convenience: functional transforms and simple arithmetic on results -----
    def _apply_to_leaves(self, fn):
        """
        Apply a function to the leaf values recursively (handles grouped nested BatchResult).
        Returns a new BatchResult preserving structure and parent collection refs.
        """
        out = {}
        for k, v in self.items():
            if isinstance(v, BatchResult):
                out[k] = v._apply_to_leaves(fn)
            else:
                out[k] = fn(v)
        return BatchResult(out, self._parent_collection)

    def map(self, fn):
        """
        Apply fn to each leaf value; fn receives the leaf value (e.g., Series/FeaturesResult) and returns a new value.
        """
        return self._apply_to_leaves(fn)

    def astype(self, dtype):
        """
        Call .astype(dtype) on each leaf Series-like value.
        """
        return self._apply_to_leaves(lambda v: v.astype(dtype))

    def _binary_op(self, other, op):
        """
        Elementwise binary op with:
          - scalar 'other'
          - another BatchResult with the same key structure
        """
        from collections.abc import Mapping

        def combine(a, b):
            # a is leaf value or BatchResult; b mirrors structure of a
            if isinstance(a, BatchResult) and isinstance(b, BatchResult):
                out = {}
                # Require identical keys to keep semantics strict
                if set(a.keys()) != set(b.keys()):
                    raise KeyError("BatchResult key mismatch in binary operation")
                for k in a.keys():
                    out[k] = combine(a[k], b[k])
                return BatchResult(out, a._parent_collection)
            if isinstance(a, BatchResult):
                # other is scalar or mapping mirroring a
                out = {}
                if isinstance(b, Mapping):
                    if set(a.keys()) != set(b.keys()):
                        raise KeyError("BatchResult key mismatch in binary operation")
                    for k in a.keys():
                        out[k] = combine(a[k], b[k])
                else:
                    for k in a.keys():
                        out[k] = combine(a[k], b)
                return BatchResult(out, a._parent_collection)
            # a is a leaf (Series/FeaturesResult)
            if isinstance(b, Mapping):
                # expect leaf value under exact key path; this case should have been handled above
                raise KeyError("Unexpected mapping at leaf in binary operation")
            return op(a, b)

        return combine(self, other)

    # Arithmetic operators
    def __add__(self, other):
        import operator

        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        import operator

        return self._binary_op(other, operator.sub)

    def __mul__(self, other):
        import operator

        return self._binary_op(other, operator.mul)

    def __truediv__(self, other):
        import operator

        return self._binary_op(other, operator.truediv)

    # Logical operators (expect boolean leaves)
    def __or__(self, other):
        import operator

        return self._binary_op(other, operator.or_)

    def __and__(self, other):
        import operator

        return self._binary_op(other, operator.and_)

    def __xor__(self, other):
        import operator

        return self._binary_op(other, operator.xor)

    def __invert__(self):
        """
        Elementwise logical NOT on leaves.
        """

        def _not(v):
            return ~v

        return self._apply_to_leaves(_not)
