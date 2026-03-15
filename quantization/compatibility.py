import pycuda.driver as cuda
import pycuda.autoinit
print(f"Device Name: {cuda.Device(0).name()}")
print(f"Compute Capability: {cuda.Device(0).compute_capability()}")