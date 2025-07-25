import onnx
onnx_model = onnx.load("C:/avetisIHA/outputs_frcnn/frcnn_mobilenetv3.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX modeli geçerli ✔️")
