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
