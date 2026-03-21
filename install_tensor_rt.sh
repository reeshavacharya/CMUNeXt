# 1. Define the path to the Python wheels in your TensorRT folder
TRT_WHEEL_DIR="/opt/TensorRT-10.15.1.29/python"

# 2. Install the three core components using the Python 3.11 (cp311) wheels
pip install $TRT_WHEEL_DIR/tensorrt-10.15.1.29-cp311-none-linux_x86_64.whl \
            $TRT_WHEEL_DIR/tensorrt_lean-10.15.1.29-cp311-none-linux_x86_64.whl \
            $TRT_WHEEL_DIR/tensorrt_dispatch-10.15.1.29-cp311-none-linux_x86_64.whl