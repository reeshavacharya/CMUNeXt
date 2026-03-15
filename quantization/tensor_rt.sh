export TENSORRT_HOME=$HOME/Downloads/TensorRT-10.15.1.29
export PATH=$TENSORRT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH

trtexec \
    --onnx=../checkpoint/quantized/CMUNeXt_model_busi_quantized.onnx \
    --int8 \
    --precisionConstraints=obey \
    --saveEngine=../checkpoint/quantized/cmunext_int8.path