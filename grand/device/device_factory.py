from cuda_device import CudaDevice

class DeviceFactory:

    @staticmethod
    def get_device(device):
        if device == "cuda":
            return CudaDevice()