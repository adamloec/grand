from . import _gcuda

print(_gcuda.cudaDeviceExists(0))

class Device:
    def __init__(self, device_type):
        self.device_type = device_type

    def allocate(self, nbytes):
        if self.device_type == 'gpu':
            pass