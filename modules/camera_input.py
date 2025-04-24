import cv2

# OpenCV ile kamera başlatılır (0: USB kamera, 1 veya 2: CSI olabilir)
cap = cv2.VideoCapture(0)

# Gerekirse çözünürlük ayarlanabilir
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_camera_frame():
    """Kameradan bir kare (frame) alır ve döner."""
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        return None
    return frame
