def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    else:
        raise ValueError("Only in-memory dictionary and Redis are supported.")
 
 
class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError
 
    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError
 
    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError
 
    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError
 
    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.
 
        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError
 
    def get_list(self, key):
        """ Returns a list stored in storage at `key`.
 
        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError
 
 
class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()
 
    def keys(self):
        return self.storage.keys()
 
    def set_val(self, key, val):
        self.storage[key] = val
 
    def get_val(self, key):
        return self.storage[key]
 
    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)
 
    def get_list(self, key):
        return self.storage.get(key, [])