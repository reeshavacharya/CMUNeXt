import onnx

model = onnx.load("../checkpoint/quantized/CMUNeXt_model_busi_quantized.onnx")
print(set(node.op_type for node in model.graph.node))