import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Engine yükle
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]

# TensorRT engine yükleniyor
engine = load_engine("frcnn_mobilenetv3.trt")
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Video kaynağı
cap = cv2.VideoCapture("test_video.mp4")  # kendi videonu buraya koy

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (416, 416))
    input_image = frame_resized.transpose((2, 0, 1)).astype(np.float32)  # CHW
    input_image = np.expand_dims(input_image, axis=0).ravel()

    inputs[0][0][:] = input_image

    start = time.time()
    output_data = do_inference(context, bindings, inputs, outputs, stream)
    end = time.time()

    print(f"Inference süresi: {(end - start) * 1000:.2f} ms")

    # Bu kısımda gerçek model çıktına göre bounding box çizimi yapılmalı
    # Bu örnek şablondur. Gerçek box koordinatlarını ve skorları output_data'dan çekmelisin.

    # Örnek: dummy bounding box çiz (düzenlenecek)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(0.3*w), int(0.3*h)), (int(0.6*w), int(0.6*h)), (0, 255, 0), 2)
    cv2.putText(frame, "Object", (int(0.3*w), int(0.3*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
