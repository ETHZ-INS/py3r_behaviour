class SummaryResult:
    def __init__(self, value, summary_obj, func_name, params, ylabel=None):
        self.value = value
        self._summary_obj = summary_obj
        self._func_name = func_name
        self._params = params
        self._ylabel = ylabel

    def store(self, name=None, meta=None, overwrite=False):
        if name is None:
            name = self._func_name
        if meta is None:
            meta = dict(self._params) if self._params else {}
        else:
            meta = dict(meta)
        # Persist ylabel so string-key lookups can recover it later
        if self._ylabel is not None:
            meta["_ylabel"] = self._ylabel
        self._summary_obj.store(self.value, name, overwrite=overwrite, meta=meta)
        return name

    def __repr__(self):
        return repr(self.value)

    def __getattr__(self, attr):
        return getattr(self.value, attr)

    def __getitem__(self, key):
        return self.value[key]
