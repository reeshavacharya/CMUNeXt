nsys profile --force-overwrite true --output=trt_profile \
    trtexec --loadEngine=../checkpoint/quantized/cmunext_int8.plan \
    --shapes=input:1x3x256x256 --verbose
nsys stats trt_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format json \
    --output trt_profile \
    --force-export true