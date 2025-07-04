class BatchProcessError(Exception):
    def __init__(self, collection_name, object_name, method, original_exception):
        self.collection_name = collection_name
        self.object_name = object_name
        self.method = method
        self.original_exception = original_exception
        super().__init__(
            f"Error in collection '{collection_name}', object '{object_name}', method '{method}': {original_exception}"
        ) 