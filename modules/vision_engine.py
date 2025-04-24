# modules/vision_engine.py
import cv2
import numpy as np

def process_frame(frame):
    """
    Gelen görüntü üzerinde renk bazlı bir maskeleme uygular,
    konturları tespit eder, merkez noktasını hesaplar.
    """
    result = {
        "detected": False,
        "center": None,
        "area": 0
    }

    # BGR'den HSV renk uzayına geç
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Belirli bir renk aralığı (örnek: mavi)
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Kontur bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # En büyük konturu bul
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 100:  # Gürültüleri engellemek için eşik
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                result.update({
                    "detected": True,
                    "center": (cx, cy),
                    "area": area
                })

    return result
