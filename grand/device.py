from . import _gmetal

class Device:
    def __init__(self, device_type):
        self.device_type = device_type

    def allocate(self, nbytes):
        if self.device_type == 'gpu':
            pass

    def add(self, a, b):
        return _gmetal.add(a, b)