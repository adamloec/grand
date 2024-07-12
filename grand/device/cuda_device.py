import numpy as np
import gcuda

class CudaDevice:
    def __init__(self):
        self.context = gcuda.CudaContext()

    def allocate(self, data):
        buffer = gcuda.CudaTensor(data.nbytes)
        buffer.copy_from_host(data)
        return buffer

    def copy_to_host(self, buffer):
        data = np.empty(buffer.size // 4, dtype=np.float32)
        gcuda.cudaMemcpyDtoH(data, buffer.get_ptr(), buffer.size)
        return data