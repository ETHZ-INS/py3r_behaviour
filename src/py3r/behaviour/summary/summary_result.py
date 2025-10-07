class SummaryResult:
    def __init__(self, value, summary_obj, func_name, params):
        self.value = value
        self._summary_obj = summary_obj
        self._func_name = func_name
        self._params = params

    def store(self, name=None, meta=None, overwrite=False):
        if name is None:
            name = self._func_name
        if meta is None:
            meta = self._params
        self._summary_obj.store(self.value, name, overwrite=overwrite, meta=meta)
        return name

    def __repr__(self):
        return repr(self.value)

    def __getattr__(self, attr):
        return getattr(self.value, attr)

    def __getitem__(self, key):
        return self.value[key]
