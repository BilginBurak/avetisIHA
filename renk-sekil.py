import cv2
import numpy as np

# Konturun şekline göre isim ve şekil skoru döndürür
def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "ucgen", 1.0
    elif num_vertices == 6:
        return "altigen", 1.0
    return "bilinmeyen", 0.0

# Renk maskesine göre kontur içindeki renk yoğunluğunu hesaplar
def color_score(mask, contour):
    mask_blank = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_blank, [contour], -1, 255, -1)
    mean_val = cv2.mean(mask, mask=mask_blank)[0]
    return min(mean_val / 255, 1.0)

# Kamerayı başlatır
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hata: Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hata: Görüntü alınamadı.")
        break

    # görüntüyü yumuşatır ve HSV'ye çevirir
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # kırmızı için hsv uzayında iki ayrı ton aralığı oluyor bu nedenle iki ayrı maske oluşturuyoruz
    #normal hsv uzayındaki değerlerin opencvdkarşılığı farklı oluyor. 
    # Hue (H):	normakalde 0-360 arasında ama opencv'de 0-179 arasında
    # Saturation (S):	normalde 0-100 arasında ama opencv'de 0-255 arasında
    # Value (V):	normalde 0-100 arasında ama opencv'de 0-255 arasında
    # https://pseudopencv.site/utilities/hsvcolormask/ sitesinden opencv'deki hsv değerlerini yüklediğiniz görsel ile kontrol edebilirsiniz
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 120, 120])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Mavi renk aralığıbelirlenyor
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # gürültü azaltma işlemleri
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # konturları bul
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # kırmızı üçgenleri tespit eder
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 50000:
            continue  # Çok küçük veya çok büyükse geç

        shape, shape_score = detect_shape(cnt)
        if shape != "ucgen":
            continue

        redness = color_score(red_mask, cnt)
        confidence = int((shape_score * 0.5 + redness * 0.5) * 100)

        if confidence > 70:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"kirmizi ucgen"
                cv2.putText(frame, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    # mavi altıgenleri tespit eder
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 50000:
            continue

        shape, shape_score = detect_shape(cnt)
        if shape != "altigen":
            continue

        blueness = color_score(blue_mask, cnt)
        confidence = int((shape_score * 0.5 + blueness * 0.5) * 100)

        if confidence > 70:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"mavi altigen"
                cv2.putText(frame, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    # Sonuçları gösteren ekran
    cv2.imshow("Canliq Tespit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()