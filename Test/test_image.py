import cv2
from modules.vision_engine import process_frame

# Görselin dosya yolunu gir
image_path = "test_images/test_image_2.jpg"

# Görseli yükle
frame = cv2.imread(image_path)
frame=cv2.resize(frame, (640, 480))
if frame is None:
    print("Görsel yüklenemedi. Dosya yolunu kontrol et.")
    exit()

# Görüntüyü işleyelim
result = process_frame(frame, area_threshold=100, debug=True)

# Sonuçları yazdır
if result["detections"]:
    for detection in result["detections"]:
        print(f"Renk: {detection['color']}")
        print(f"Şekil: {detection['shape']}")
        print(f"Merkez: {detection['center']}")
        print(f"Alan: {detection['area']}")
else:
    print("Hiçbir nesne algılanmadı.")


# Sonucu görselleştir
cv2.imshow("Test Görseli", result["annotated_frame"])
cv2.waitKey(0)
cv2.destroyAllWindows()
