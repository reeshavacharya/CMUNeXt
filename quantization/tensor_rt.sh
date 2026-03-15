export TENSORRT_HOME=$HOME/Downloads/TensorRT-10.15.1.29
export PATH=$TENSORRT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH

trtexec \
    --onnx=../checkpoint/CMUNeXt_model_busi_quantized.onnx \
    --int8 \
    --precisionConstraints=obey \
    --exportLayerInfo=engine_info.json \
    --saveEngine=../checkpoint/quantized/cmunext_int8.plan