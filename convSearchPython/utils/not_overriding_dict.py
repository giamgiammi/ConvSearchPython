"""Contains class for a dict that not allow overriding already existing value"""


class NotOverridingDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f'key {key} already present')
        return super().__setitem__(key, value)
