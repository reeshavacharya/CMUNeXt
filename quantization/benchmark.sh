export TENSORRT_HOME=$HOME/Downloads/TensorRT-10.15.1.29
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH

LD_PRELOAD=$TENSORRT_HOME/lib/libnvinfer.so.10 python3 benchmark.py