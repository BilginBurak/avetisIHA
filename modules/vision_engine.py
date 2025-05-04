import cv2
import numpy as np
from modules.filtering import preprocess_frame, get_dynamic_kernel, apply_morphological_filters, filter_contour

COLOR_RANGES = {
    "blue": {
        "lower": np.array([100, 150, 0]),
        "upper": np.array([140, 255, 255])
    },
    "red": [
        {"lower": np.array([0, 100, 100]), "upper": np.array([20, 255, 255])},
        {"lower": np.array([160, 100, 100]), "upper": np.array([180, 255, 255])}
    ]
}


def detect_shape(approx):
    """
    Köşe sayısına göre konturun üçgen mi yoksa altıgen mi olduğunu algılar.

    Args:
        approx (np.ndarray): YAklaşık kontur noktaları.

    Returns:
        str or None:  3 küşe için 'triangle',  5-7 köşe için 'hexagon', Değilse None.
    """
    vertices = len(approx)
    if vertices == 3:
        return "triangle"
    elif 5 <= vertices <= 7:
        return "hexagon"
    return None


def process_frame(frame, area_threshold=500, debug=True):
    """
    filtrelemeyle mavi altıgenleri ve kırmızı üçgenleri algılamak için bir çerçeveyi işler.

    Args:
        frame (np.ndarray): BGR resim.
        area_threshold (float): Algılama için minimum kontur alanı.
        debug (bool): True ise görselleştirmelerle açıklamalı çerçeveyi döndürür.


    Returns:
        List: tespit edilenlerin listesi, hepsi  anahatarlı bir szölük yapısında saklanır:
            - center (tuple): (cx, cy) Ağırlık noktasının kordinatı.
            - area (float): Kontur alanı.
            - shape (str): Algılanan şekil ('triangle' or 'hexagon') değilse None.
            - color (str): Algılanan Renk ('blue' or 'red') or None.
            - annotated_frame (np.ndarray): Çizilmiş konturlar, ağırlık merkezi ve etiket içeren çerçeve eğer debug='True' ise
    """
    detections = []
    annotated_frame = frame.copy() if debug else None

    # Preprocess frame
    frame = preprocess_frame(frame, blur_kernel_size=3)

    # HSV dünüşümü
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dinamik kernel alma
    img_area = frame.shape[0] * frame.shape[1]
    kernel = get_dynamic_kernel(img_area,)

    # Mavi ve kırmızı rengi işleme
    for color_name in COLOR_RANGES:
        # Kırmızı için geniş tanımlama
        ranges = COLOR_RANGES[color_name] if isinstance(COLOR_RANGES[color_name], list) else [COLOR_RANGES[color_name]]
        for color_range in ranges:
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])

            # morphological filtresi uygula
            mask = apply_morphological_filters(mask, kernel, open_iterations=1, erode_iterations=1, dilate_iterations=1)

            # konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            for cnt in contours:
                # alana ve daireselliğe göre Konturları filtrele
                if not filter_contour(cnt, area_threshold, min_circularity=0.5):
                    continue

                # Yaklaşık konturlar
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Şekil doğrulaması
                shape = detect_shape(approx)
                if shape and ((color_name == "blue" and shape == "hexagon") or
                              (color_name == "red" and shape == "triangle")):
                    # Ağırlık merkezi hesaplama
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # resultu güncelleme
                    detection = {
                        "center": (cx, cy),
                        "area": cv2.contourArea(cnt),
                        "shape": shape,
                        "color": color_name
                    }
                    detections.append(detection)

                    # çerçeveyi oluşturma
                    if debug:
                        cv2.drawContours(annotated_frame, [cnt], -1, (0, 255, 0), 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                        label = f"{color_name} {shape}"
                        cv2.putText(annotated_frame, label, (cx - 50, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return {"detections": detections, "annotated_frame": annotated_frame}