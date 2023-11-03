from numba import cuda

class Device:
    def __init__(self, device_type):
        self.device_type = device_type
        self.data = None
    
    def move_to(self, device_type):
        pass
