# Set CUDA_HOME to /usr (since nvcc is in /usr/bin)
export CUDA_HOME=/usr
export PATH=$PATH:/usr/bin
# Add the common library path for Ubuntu-installed CUDA
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Also add the TensorRT path to be safe
export TENSORRT_LIB=$HOME/Downloads/TensorRT-10.15.1.29/lib
export LD_LIBRARY_PATH=$TENSORRT_LIB:$LD_LIBRARY_PATH

python3 onnx_export.py