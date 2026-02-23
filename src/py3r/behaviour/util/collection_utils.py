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

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def _leaf_count(d):
        """Count total leaf entries and how many are None, recursively."""
        total = 0
        none_count = 0
        for v in d.values():
            if isinstance(v, dict):
                t, n = BatchResult._leaf_count(v)
                total += t
                none_count += n
            else:
                total += 1
                if v is None:
                    none_count += 1
        return total, none_count

    @staticmethod
    def _leaf_type_name(d):
        """Return the type name of the first non-None leaf, or None."""
        for v in d.values():
            if isinstance(v, dict):
                name = BatchResult._leaf_type_name(v)
                if name is not None:
                    return name
            elif v is not None:
                return type(v).__name__
        return None

    def __repr__(self):
        total, none_count = self._leaf_count(self)
        nested_groups = [k for k, v in self.items() if isinstance(v, dict)]

        # All None -> compact in-place summary
        if none_count == total and not nested_groups:
            return f"BatchResult: {total} items processed (in-place)"

        # No Nones and small enough -> show abbreviated dict
        type_name = self._leaf_type_name(self) or "?"
        if nested_groups:
            group_lines = []
            for gk in self:
                sub = self[gk]
                if isinstance(sub, dict):
                    t, n = self._leaf_count(sub)
                    if n == t:
                        group_lines.append(f"  {gk}: {t} items (in-place)")
                    else:
                        group_lines.append(f"  {gk}: {t} items -> {type_name}")
                else:
                    val_str = "None" if sub is None else type(sub).__name__
                    group_lines.append(f"  {gk}: {val_str}")
            body = "\n".join(group_lines)
            return f"BatchResult ({total} items, {len(self)} groups):\n{body}"

        # Flat structure -- show keys with truncated values
        if total <= 8:
            lines = []
            for k, v in self.items():
                if v is None:
                    lines.append(f"  {k}: None (in-place)")
                else:
                    v_repr = repr(v)
                    if len(v_repr) > 80:
                        v_repr = v_repr[:77] + "..."
                    lines.append(f"  {k}: {v_repr}")
            body = "\n".join(lines)
            return f"BatchResult ({total} items -> {type_name}):\n{body}"

        # Large flat -- just summarise
        if none_count == 0:
            return f"BatchResult: {total} items -> {type_name}"
        return (
            f"BatchResult: {total} items "
            f"({total - none_count} -> {type_name}, {none_count} in-place)"
        )

    def plot(self, *args, **kwargs):
        return self._parent_collection.plot(self, *args, **kwargs)

    def store(self, *args, **kwargs):
        return self._parent_collection.store(self, *args, **kwargs)

    # ----- Convenience: functional transforms and simple arithmetic on results -----
    def _apply_to_leaves(self, fn):
        """
        Apply a function to the leaf values recursively (handles grouped
        nested BatchResult). Returns a new BatchResult preserving
        structure and parent collection refs.
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
        Apply fn to each leaf value; fn receives the leaf value
        (e.g., Series/FeaturesResult) and returns a new value.
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
                # expect leaf value under exact key path; handled above
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

    # Reflected arithmetic operators
    def __radd__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.add(b, a))

    def __rsub__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.sub(b, a))

    def __rmul__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.mul(b, a))

    def __rtruediv__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.truediv(b, a))

    # Comparisons
    def __lt__(self, other):
        import operator

        return self._binary_op(other, operator.lt)

    def __le__(self, other):
        import operator

        return self._binary_op(other, operator.le)

    def __gt__(self, other):
        import operator

        return self._binary_op(other, operator.gt)

    def __ge__(self, other):
        import operator

        return self._binary_op(other, operator.ge)

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

    # Reflected logical operators (for scalar bool on the left)
    def __ror__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.or_(b, a))

    def __rand__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.and_(b, a))

    def __rxor__(self, other):
        import operator

        return self._binary_op(other, lambda a, b: operator.xor(b, a))

    def __invert__(self):
        """
        Elementwise logical NOT on leaves.
        """

        def _not(v):
            return ~v

        return self._apply_to_leaves(_not)
